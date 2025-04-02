from functools import partial
import torch
from torch import nn
from copy import deepcopy

from utils.colors import *
from utils.dist_utills import ddp_is_on, print_ddp
from .loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from .layers import DINOHead
from .dinov2_utils import has_batchnorms
from utils import helpfuns
from utils.colors import *
from . import dinov2_vits as vits

import warnings
# Suppress the specific warning related to xFormers
warnings.filterwarnings('ignore', message=".*xFormers is available.*")

from apla.apla_vit import build_apla


try:
    from xformers.ops import fmha
except ImportError:
    raise AssertionError("xFormers is required for training")


LVD_142M_STATES = {
    'vit_small': 'dinov2_vits14',
    'vit_base': 'dinov2_vitb14',
    'vit_large': 'dinov2_vitl14',
    'vit_giant': 'dinov2_vitg14'
}


def build_model(backbone, backbone_args, only_teacher=False, img_size=224):
    vit_kwargs = dict(
        img_size=img_size,
        patch_size=backbone_args.patch_size,
        init_values=backbone_args.layerscale,
        ffn_layer=backbone_args.ffn_layer,
        block_chunks=backbone_args.block_chunks,
        num_register_tokens=backbone_args.num_register_tokens,
        interpolate_offset=backbone_args.interpolate_offset,
        interpolate_antialias=backbone_args.interpolate_antialias,
    )
    if only_teacher:
        raise NotImplementedError
    teacher = vits.__dict__[backbone](**vit_kwargs)
    student = vits.__dict__[backbone](
        **vit_kwargs,
        drop_path_rate=backbone_args.drop_path_rate,
        drop_path_uniform=backbone_args.drop_path_uniform,
    )
    print_ddp(blue(f'Created dinov2 ranomly init student and teacher and returning them'))
    embed_dim = student.embed_dim
    return student, teacher, embed_dim


class DINOv2(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.wrapper_params = params
        # --- helper variables
        dino_params = self.wrapper_params.model_params.dinov2.dino
        ibot_params = self.wrapper_params.model_params.dinov2.ibot
        self.model_params = self.wrapper_params.model_params
        student_params = self.wrapper_params.model_params.transformers_params.student
        
        # --- init and load studet/teacher
        student_model_dict = dict()
        teacher_model_dict = dict()

        assert 'vit' in self.model_params.backbone_type, 'Only supports ViT'
        assert student_params.pretrained_type == 'LVD142M-SSL', 'Only supports LVD142M-SSL pretraining'
        
        student_backbone, teacher_backbone, embed_dim = build_model(backbone=self.model_params.backbone_type,
                                                                    backbone_args=student_params,
                                                                    only_teacher=False,
                                                                    img_size=student_params.pre_img_size)
        # load pretrained network weights
        if params.model_params.pretrained:
            dinov2_state = torch.hub.load(repo_or_dir='facebookresearch/dinov2', 
                                          model=LVD_142M_STATES[self.model_params.backbone_type]).state_dict()
            print_ddp(cyan(f'Initializing Dinov2 -- Using LVD142M-SSL states'))
            
            missing_keys, unexpected_keys = student_backbone.load_state_dict(dinov2_state, strict=False)
            assert (missing_keys == [] or missing_keys == ['mask_token']) and (unexpected_keys == [])
            teacher_backbone.load_state_dict(deepcopy(student_backbone.state_dict()))
            
            print_ddp(cyan(f'Initializing Dinov2 -- missing_keys: {missing_keys}'))
            print_ddp(cyan(f'Initializing Dinov2 -- unexpected_keys: {unexpected_keys}'))
            print_ddp(cyan(f'Initializing Dinov2 -- Successfully loaded weights into student and teacher'))
        
        if 'adaptation' in self.model_params:
            adaptation_conf = self.model_params.adaptation
            assert adaptation_conf.mode == 'apla', 'Only supports adaptation with APLA'
            print_ddp(f'Initializing Dinov2 -- Enabling adaptation with: {adaptation_conf.mode}')
            
            print_ddp(blue(f'Initializing Dinov2 -- Building Student APLA'))
            student_backbone = build_apla(
                config=adaptation_conf.params,
                model=student_backbone,
                attn_class='apla_attn_mem_eff',
                is_multi_gpu=len(self.wrapper_params.system_params.which_GPUs.split(",")) > 1,
            )
            
            print_ddp(blue(f'Initializing Dinov2 -- Building Teacher APLA'))
            teacher_backbone = build_apla(
                config=adaptation_conf.params,
                model=teacher_backbone,
                attn_class='apla_attn_mem_eff',
                is_multi_gpu=len(self.wrapper_params.system_params.which_GPUs.split(",")) > 1,
            )
            for name, param in teacher_backbone.named_parameters():
                param.requires_grad = False
            print_ddp(gray(f'Initializing Dinov2 -- Freezed all Teacher parameters'))
        
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone

        # --- init dino/ibot heads
        self.do_dino = dino_params.loss_weight > 0
        self.do_koleo = dino_params.koleo_loss_weight > 0
        self.do_ibot = ibot_params.loss_weight > 0
        self.ibot_separate_head = ibot_params.separate_head

        if self.do_dino or self.do_ibot:
            dino_head = partial(
                DINOHead,
                in_dim=embed_dim,
                out_dim=dino_params.head_n_prototypes,
                hidden_dim=dino_params.head_hidden_dim,
                bottleneck_dim=dino_params.head_bottleneck_dim,
                nlayers=dino_params.head_nlayers,
            )
            
        if self.do_dino:
            self.dino_loss_weight = dino_params.loss_weight
            self.dino_loss = DINOLoss(out_dim=dino_params.head_n_prototypes)
            if self.do_koleo:
                self.koleo_loss = KoLeoLoss()
        else:
            pass

        if self.do_dino or self.do_ibot:
            student_model_dict["dino_head"] = dino_head()
            teacher_model_dict["dino_head"] = dino_head()

        if self.do_ibot:
            self.ibot_loss_weight = ibot_params.loss_weight
            assert max(ibot_params.mask_ratio_min_max) > 0, "please provide a positive mask ratio tuple for ibot"
            assert ibot_params.mask_sample_probability > 0, "please provide a positive mask probability for ibot"
            self.ibot_out_dim = ibot_params.head_n_prototypes if self.ibot_separate_head else dino_params.head_n_prototypes
            self.ibot_patch_loss = iBOTPatchLoss(patch_out_dim=self.ibot_out_dim)
            
            if self.ibot_separate_head:
                ibot_head = partial(
                    DINOHead,
                    in_dim=embed_dim,
                    out_dim=params.ibot.head_n_prototypes,
                    hidden_dim=params.ibot.head_hidden_dim,
                    bottleneck_dim=params.ibot.head_bottleneck_dim,
                    nlayers=params.ibot.head_nlayers,
                )
                student_model_dict["ibot_head"] = ibot_head()
                teacher_model_dict["ibot_head"] = ibot_head()
            else:
                pass

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        
        # load student weights into teacher
        for k, v in self.student.items():
            self.teacher[k].load_state_dict(self.student[k].state_dict())
        print_ddp(gray('Initializing Dinov2 -- Loading all student states into teacher: done'))
            
        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
        print_ddp(gray('Initializing Dinov2 -- Setting teacher params to no grad: done'))
        
        print_ddp(blue(f'Initializing Dinov2 -- Objective: do_dino: {self.do_dino} -- do_ibot: {self.do_ibot}'))
        print_ddp(blue(
            f'Initializing Dinov2 -- Model total params: {helpfuns.count_total_params(self):,} '
            f'-- model trainable params: {helpfuns.count_trainable_params(self):,}'
        ))
        print_ddp(blue(
            f'Initializing Dinov2 -- TEACHER Model total params: {helpfuns.count_total_params(self.teacher):,} -- '
            f'backbone: {helpfuns.count_total_params(self.teacher["backbone"]):,} '
            f'dino_head: {helpfuns.count_total_params(self.teacher["dino_head"]):,} '
            f'-- model trainable params: {helpfuns.count_trainable_params(self.teacher):,}'
        ))
        print_ddp(blue(
            f'Initializing Dinov2 -- STUDENT Model total params: {helpfuns.count_total_params(self.student):,} -- '
            f'backbone: {helpfuns.count_total_params(self.student["backbone"]):,} '
            f'dino_head: {helpfuns.count_total_params(self.student["dino_head"]):,} '
            f'-- model trainable params: {helpfuns.count_trainable_params(self.student):,} '
            f'backbone: {helpfuns.count_trainable_params(self.student["backbone"]):,} '
            f'dino_head: {helpfuns.count_trainable_params(self.student["dino_head"]):,}'
            f'\n'
        ))
        # print(self.student.keys())

    def forward(self, images, return_embedding=False, teacher_temp=None):
        if return_embedding:  # only for evaluation purposes
            images = images.cuda(non_blocking=True)
            return [None, None], self.teacher.backbone.forward_features(images)["x_norm_clstoken"]  # images is simply a batch of images
        
        n_global_crops = self.wrapper_params.crops_params.n_global_crops
        n_local_crops = self.wrapper_params.crops_params.n_local_crops
        assert n_global_crops == 2

        global_crops = images["collated_global_crops"].cuda(non_blocking=True)
        local_crops = images["collated_local_crops"].cuda(non_blocking=True)

        masks = images["collated_masks"].cuda(non_blocking=True)
        mask_indices_list = images["mask_indices_list"].cuda(non_blocking=True)
        n_masked_patches_tensor = images["n_masked_patches"].cuda(non_blocking=True)
        n_masked_patches = mask_indices_list.shape[0]
        upperbound = images["upperbound"]
        masks_weight = images["masks_weight"].cuda(non_blocking=True)

        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
        n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops

        do_dino = self.do_dino
        do_ibot = self.do_ibot
        
        # loss scales
        ibot_loss_scale = 1.0 / n_global_crops
        
        # teacher output
        @torch.no_grad()
        def get_teacher_output():
            x, n_global_crops_teacher = global_crops, n_global_crops
            teacher_backbone_output_dict = self.teacher.backbone(x, is_training=True)
            teacher_cls_tokens = teacher_backbone_output_dict["x_norm_clstoken"]
            teacher_cls_tokens = teacher_cls_tokens.chunk(n_global_crops_teacher)
            # watch out: these are chunked and cat'd in reverse so A is matched to B in the global crops dino loss
            teacher_cls_tokens = torch.cat((teacher_cls_tokens[1], teacher_cls_tokens[0]))
            ibot_teacher_patch_tokens = teacher_backbone_output_dict["x_norm_patchtokens"]
            _dim = ibot_teacher_patch_tokens.shape[-1]
            n_cls_tokens = teacher_cls_tokens.shape[0]

            if do_ibot and not self.ibot_separate_head:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound + n_cls_tokens, _dim)
                buffer_tensor_teacher[:n_cls_tokens].copy_(teacher_cls_tokens)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[n_cls_tokens : n_cls_tokens + n_masked_patches],
                )
                tokens_after_head = self.teacher.dino_head(buffer_tensor_teacher)
                teacher_cls_tokens_after_head = tokens_after_head[:n_cls_tokens]
                masked_teacher_patch_tokens_after_head = tokens_after_head[
                    n_cls_tokens : n_cls_tokens + n_masked_patches
                ]
            elif do_ibot and self.ibot_separate_head:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound, _dim)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[:n_masked_patches],
                )
                teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
                masked_teacher_patch_tokens_after_head = self.teacher.ibot_head(buffer_tensor_teacher)[
                    :n_masked_patches
                ]
            else:
                teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
                masked_teacher_ibot_softmaxed_centered = None

            # centering
            teacher_dino_softmaxed_centered_list = None
            masked_teacher_ibot_softmaxed_centered = None
            if self.model_params.dinov2.centering == "centering":
                if do_dino:
                    teacher_dino_softmaxed_centered_list = self.dino_loss.softmax_center_teacher(
                        teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                    ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])
                    self.dino_loss.update_center(teacher_cls_tokens_after_head)
                if do_ibot:
                    masked_teacher_patch_tokens_after_head = masked_teacher_patch_tokens_after_head.unsqueeze(0)
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.softmax_center_teacher(
                        masked_teacher_patch_tokens_after_head[:, :n_masked_patches], teacher_temp=teacher_temp
                    )
                    masked_teacher_ibot_softmaxed_centered = masked_teacher_ibot_softmaxed_centered.squeeze(0)
                    self.ibot_patch_loss.update_center(masked_teacher_patch_tokens_after_head[:n_masked_patches])

            elif self.model_params.dinov2.centering == "sinkhorn_knopp":
                if do_dino:
                    teacher_dino_softmaxed_centered_list = self.dino_loss.sinkhorn_knopp_teacher(
                        teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                    ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])

                if do_ibot:
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
                        masked_teacher_patch_tokens_after_head,
                        teacher_temp=teacher_temp,
                        n_masked_patches_tensor=n_masked_patches_tensor,
                    )

            else:
                raise NotImplementedError

            return teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered

        
        teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered = get_teacher_output()
        # reshard_fsdp_model(self.teacher)

        loss_dict = {}

        loss_accumulator = 0  # for backprop
        student_global_backbone_output_dict, student_local_backbone_output_dict = self.student.backbone(
            [global_crops, local_crops], masks=[masks, None], is_training=True
        )

        inputs_for_student_head_list = []

        # --- shapes
        # student_local_backbone_output_dict["x_norm_clstoken"]: (512, 384) -- 512: batch size (64) * n_local_crops (8)
        # student_local_backbone_output_dict['x_norm_patchtokens']: (512, 49, 384) -- 49: (98/14) ** 2
        #
        # student_global_cls_tokens: (128, 384) -- 128: batch size (64) * n_global_crops (2)
        # student_global_backbone_output_dict['x_norm_patchtokens']: (128, 256, 384) -- 256: (224/14) ** 2
        # ---
        
        # 1a: local crops cls tokens
        student_local_cls_tokens = student_local_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list.append(student_local_cls_tokens.unsqueeze(0))

        # 1b: global crops cls tokens
        student_global_cls_tokens = student_global_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list.append(student_global_cls_tokens.unsqueeze(0))

        # 1c: global crops patch tokens
        if do_ibot:
            _dim = student_global_backbone_output_dict["x_norm_clstoken"].shape[-1]
            ibot_student_patch_tokens = student_global_backbone_output_dict["x_norm_patchtokens"]
            buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(upperbound, _dim)
            buffer_tensor_patch_tokens[:n_masked_patches].copy_(
                torch.index_select(ibot_student_patch_tokens.flatten(0, 1), dim=0, index=mask_indices_list)
            )
            if not self.ibot_separate_head:
                inputs_for_student_head_list.append(buffer_tensor_patch_tokens.unsqueeze(0))
            else:
                student_global_masked_patch_tokens_after_head = self.student.ibot_head(buffer_tensor_patch_tokens)[
                    :n_masked_patches
                ]

        # 2: run
        _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(inputs_for_student_head_list)
        outputs_list = _attn_bias.split(self.student.dino_head(cat_inputs))

        # 3a: local crops cls tokens
        student_local_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3b: global crops cls tokens
        student_global_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3c: global crops patch tokens
        if do_ibot and not self.ibot_separate_head:
            student_global_masked_patch_tokens_after_head = outputs_list.pop(0).squeeze(0)[:n_masked_patches]

        # local to global correspondence
        if do_dino and n_local_crops > 0:
            # ---
            # teacher_dino_softmaxed_centered_list: [2, 64, 65536], 2 global crops
            # student_local_cls_tokens_after_head: [512, 65536] -- will be chunked into 8 tensors of shape [64, 65536]
            # ---
            dino_local_crops_loss = self.dino_loss(
                student_output_list=student_local_cls_tokens_after_head.chunk(n_local_crops),
                teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list,
            ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            # display and accumulate
            loss_dict["dino_local_crops_loss"] = dino_local_crops_loss
            loss_accumulator += self.dino_loss_weight * dino_local_crops_loss

        # process global crops (global to global correspondence)
        loss_scales = 2  # this is here since we process global crops together
        if do_dino:
            # compute loss
            dino_global_crops_loss = (
                self.dino_loss(
                    student_output_list=[student_global_cls_tokens_after_head],  # list of one tensor with shape [128, 65536]
                    teacher_out_softmaxed_centered_list=[
                        teacher_dino_softmaxed_centered_list.flatten(0, 1)  # list of one tensor with shape  [128, 65536]
                    ],  # these were chunked and stacked in reverse so A is matched to B
                )
                * loss_scales
                / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            )
            # display and accumulate
            loss_dict["dino_global_crops_loss"] = dino_global_crops_loss
            loss_accumulator += self.dino_loss_weight * dino_global_crops_loss

            # --- koleo
            # student_cls_tokens: (128, 384)
            # student_cls_tokens.chunk(2) returns two chunks of shapes (64, 384)
            # corresponding to the first and second global crops for each image in the batch
            # then koleo is performed on each chunk separately and then summed
            # ---
            student_cls_tokens = student_global_cls_tokens
            if self.do_koleo:
                koleo_loss = self.model_params.dinov2.dino.koleo_loss_weight * sum(
                    self.koleo_loss(p) for p in student_cls_tokens.chunk(2)
                )  # we don't apply koleo loss between cls tokens of a same image
                # display and accumulate
                loss_accumulator += koleo_loss
                loss_dict["koleo_loss"] = (
                    koleo_loss / loss_scales
                )  # this is to display the same losses as before but we can remove eventually

        if do_ibot:
            # compute loss
            ibot_patch_loss = (
                self.ibot_patch_loss.forward_masked(
                    student_global_masked_patch_tokens_after_head,  # [4884, 65536]
                    masked_teacher_ibot_softmaxed_centered,  # [4884, 65536]
                    student_masks_flat=masks,  # [128, 256]
                    n_masked_patches=n_masked_patches,  # [4884]
                    masks_weight=masks_weight,  # [4884]
                )
                * loss_scales
                * ibot_loss_scale
            )

            # store for display
            loss_dict["ibot_loss"] = ibot_patch_loss / 2

            # accumulate loss
            loss_accumulator += self.ibot_loss_weight * ibot_patch_loss

        # print(f'loss_dict: {loss_dict}')
        return loss_accumulator, loss_dict

    def update_teacher(self, m):
        student_param_list = []
        teacher_param_list = []
        with torch.no_grad():
            for k in self.student.keys():
                # for ms, mt in zip(get_fsdp_modules(self.student[k]), get_fsdp_modules(self.teacher[k])):
                for ms, mt in zip(self.student[k].parameters(), self.teacher[k].parameters()):
                    student_param_list += ms.detach()
                    teacher_param_list += mt
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)
    
    def train(self, train_mode=True):
        if train_mode:
            super().train()
        else:
            super().train(False)
        self.teacher.eval()
