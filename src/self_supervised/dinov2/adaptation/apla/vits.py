# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable
from easydict import EasyDict as edict

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from ...layers import Mlp, PatchEmbed, SwiGLUFFNFused  # , MemEffAttention, NestedTensorBlock as Block
from .block import APLA_NestedTensorBlock as Block
from .attention import APLA_MemEffAttention

from utils import helpfuns
from utils.colors import *
from utils.transformers.transformers_utils import download_weights
from utils.dist_utills import print_ddp
from copy import deepcopy
from .attention import attn_has, mlp_has


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class APLA_DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        partial_conf,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        print(f'drop_path_uniform: {drop_path_uniform}')
        print(f'drop_path_rate: {drop_path_rate}')
        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
            
        self.partial_conf = partial_conf

        if ffn_layer == "mlp":
            # logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            # logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            # logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                partial_conf=edict(
                    partial_size=self.partial_conf.partial_size[i] \
                        if 'progressive' in self.partial_conf.fastadapt_mode or self.partial_conf.ind_mode in ['v4', 'v4_2', 'v5'] \
                        else self.partial_conf.partial_size,
                    mode=self.partial_conf.mode,
                    ind_mode=self.partial_conf.ind_mode,
                ),
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)
        
        self.register_proj_indices()

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()
    
    def register_proj_indices(self):
        # mode 1: sample once, all blocks use the same set of indices
        # v2: for each block sample in the beginning
        # v3: for each block, sample in each iteration
        # v4: samples different number of rows per block, and only tunes them
        # v5: sample different number of rows and different rows per iteration
        if hasattr(self.partial_conf, 'inds_path'):
            inds_path = self.partial_conf.inds_path
            print_ddp(cyan(f'--- Registering inds based on path: {inds_path}'))
            inds_dict = helpfuns.load_json(inds_path)
            for i, block in enumerate(self.blocks):
                inds = inds_dict[f'block_{i}']
                remaining = [idx for idx in range(self.num_features) if idx not in inds]
                inds = torch.tensor(inds + remaining)  # all indices, training ones in the beginning (for later loading and training)
                block.attn.register_buffer('indices', inds)
                print_ddp(f'inds for block {i}: {inds[:10]} -- shape: {inds.shape}')
            
        else:
            ind_mode = self.partial_conf.ind_mode
            if ind_mode == 'v1':
                inds = torch.randperm(self.num_features)
                # inds = torch.arange(self.num_features)
                for i, block in enumerate(self.blocks):
                    block.attn.register_buffer('indices', inds)
                    print_ddp(f'inds for block {i}: {inds} -- shape: {inds.shape}')
            else:
                for i, block in enumerate(self.blocks):  # take random per block
                    inds = torch.randperm(self.num_features)
                    # inds = torch.arange(self.num_features)
                    block.attn.register_buffer('indices', inds)
                    print_ddp(f'For block {i} -- partial_size: {block.attn.partial_size} -- inds: {inds[:10]} -- shape: {inds.shape}')
                    # input


    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x
    
    def prepare_tokens_with_masks_add_cls(self, x, masks=None):
        # B, nc, w, h = x.shape
        B, N, C = x.shape
        w = h = self.patch_size * int(math.sqrt(N))  # original image size
        # x = self.patch_embed(x)
        if masks is not None:
            raise NotImplementedError
            # x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        orig_dtype = self.cls_token.dtype
        x = torch.index_add(input=x, dim=1, index=torch.tensor([0], device=x.device), source=self.cls_token.expand(B, -1, -1).to(x.dtype))
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = x.to(orig_dtype)

        if self.register_tokens is not None:
            raise NotImplementedError
        return x
    
    def forward_features_list(self, x_list, masks_list):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None, sum_cls=False):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        if sum_cls:
            x = self.prepare_tokens_with_masks_add_cls(x, masks)
        else:
            x = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, return_cls_patch_tokens=False,  **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:  # for SSL training
            return ret
        elif return_cls_patch_tokens:
            return torch.cat((ret["x_norm_clstoken"].unsqueeze(dim=1), ret["x_norm_patchtokens"]), dim=1)
        else:  # for supervised training/evaluation (default)
            return self.head(ret["x_norm_clstoken"])


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)



# NOTE: I just renamed vit to deit here to be consistent with the rest of the code
def deit_small(patch_size=16, num_register_tokens=0, **kwargs):
    model = APLA_DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=APLA_MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def deit_base(partial_conf, patch_size=16, num_register_tokens=0, **kwargs):
    model = APLA_DinoVisionTransformer(
        partial_conf,
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=APLA_MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def deit_large(patch_size=16, num_register_tokens=0, **kwargs):
    model = APLA_DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=APLA_MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def deit_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = APLA_DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=APLA_MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model

def get_progressive_partial_size(key, config):
    blk_ind = int(key.split('.')[1])
    partial_size = config.partial_size[blk_ind] \
        if 'progressive' in config.fastadapt_mode or config.ind_mode in ['v4', 'v4_2', 'v5'] \
            else config.partial_size
    # print_ddp(f'key: {key} -- BLK IND: {blk_ind} -- partial_size: {partial_size}')
    # input()
    return partial_size


def build_apla__(config, backbone_type, transformers_params, checkpoint, net_mode):
    model = globals()[backbone_type](partial_conf=config, **transformers_params)
    DIM = model.width if 'open_clip' in backbone_type else model.num_features
    print_ddp(f'DIM: {DIM}')
    mode = config.mode
    
    # if pretrained:
    #     checkpoint = download_weights(backbone_type=backbone_type, 
    #                                   patch_size=transformers_params.patch_size,
    #                                   pretrained_type=transformers_params.pretrained_type)
    # input()
    
    for key, _ in deepcopy(checkpoint).items():
        with torch.no_grad():
            # if 'attn.qkv' in key:  #  or 'proj.weight' in key:
            #     partial_size = get_progressive_partial_size(key, config)
                
            #     qkv = checkpoint.pop(key)
            #     q, k, v = torch.split(qkv, DIM, dim=0)
                
            #     if attn_has(mode, 'q'):
            #         q1, q2 = torch.split(q, [partial_size, DIM - partial_size], dim=0)
            #         checkpoint[key.replace('qkv', 'q1')] = q1
            #         checkpoint[key.replace('qkv', 'q2')] = q2
            #     else:
            #         checkpoint[key.replace('qkv', 'q')] = q
                
            #     if attn_has(mode, 'k'):
            #         k1, k2 = torch.split(k, [partial_size, DIM - partial_size], dim=0)
            #         checkpoint[key.replace('qkv', 'k1')] = k1
            #         checkpoint[key.replace('qkv', 'k2')] = k2
            #     else:
            #         checkpoint[key.replace('qkv', 'k')] = k
                
            #     if attn_has(mode, 'v'):
            #         v1, v2 = torch.split(v, [partial_size, DIM - partial_size], dim=0)
            #         checkpoint[key.replace('qkv', 'v1')] = v1
            #         checkpoint[key.replace('qkv', 'v2')] = v2
            #     else:
            #         checkpoint[key.replace('qkv', 'v')] = v
            
            if 'attn.proj' in key:
                partial_size = get_progressive_partial_size(key, config)
                proj = checkpoint.pop(key)
                if attn_has(mode, 'proj'):
                    i = int(key.split('.')[1])
                    inds = model.blocks[i].attn.indices  # specific indices corresponding to block i
                    # print_ddp(f'i: {i} -- inds: {inds[:10]}')
                    # input()
                    if 'weight' in key:
                        proj1 = proj[inds[:partial_size], :]
                        proj2 = proj[inds[partial_size:], :]
                        checkpoint[key.replace('proj.weight', 'proj_weight1')] = proj1
                        checkpoint[key.replace('proj.weight', 'proj_weight2')] = proj2
                    elif 'bias' in key:
                        proj1 = proj[inds[:partial_size]]  # one dimensional
                        proj2 = proj[inds[partial_size:]]
                        checkpoint[key.replace('proj.bias', 'proj_bias1')] = proj1
                        checkpoint[key.replace('proj.bias', 'proj_bias2')] = proj2
                    else:
                        raise ValueError
                else:
                    checkpoint[key.replace('proj', 'proj')] = proj
            
            elif 'mlp' in key:  # blocks.1.mlp.fc1.weight
                partial_size = get_progressive_partial_size(key, config)
                which_fc = key.split('.')[3]
                # print_ddp(f'key: {key} -- which fc: {which_fc}')
                # input()
                if mlp_has(mode, which_fc):
                    fc = checkpoint.pop(key)
                    dim = fc.shape[0]
                    fc_part_1, fc_part_2 = torch.split(fc, [partial_size, dim - partial_size], dim=0)
                    checkpoint[key.replace(which_fc, f'{which_fc}_1')] = fc_part_1
                    checkpoint[key.replace(which_fc, f'{which_fc}_2')] = fc_part_2
                        
    print_ddp(f'MODIFIED QKV TO INDIVIDUAL Q, K, V, Proj')
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    assert unexpected_keys == [], f'There are unexpected keys!: \n{unexpected_keys}\n'
    assert all(['indices' in key or key == 'mask_token' for key in missing_keys])

    trainable_parts = []
    if attn_has(mode, 'q'):
        trainable_parts.append('q1')
    if attn_has(mode, 'k'):
        trainable_parts.append('k1')
    if attn_has(mode, 'v'):
        trainable_parts.append('v1')
    if attn_has(mode, 'proj'):
        trainable_parts.append('proj_weight1')
        trainable_parts.append('proj_bias1')
    if mlp_has(mode, 'fc1'):
        trainable_parts.append(f'fc1_1')
    if mlp_has(mode, 'fc2'):
        trainable_parts.append(f'fc2_1')
    trainable_names = [name for name, _ in model.named_parameters() if any([n in name for n in trainable_parts])]
    
    print_ddp(f'TRAINABLE PARTS: {trainable_parts}')
    print_ddp(f'TRAINABLE NAMES: {trainable_names}')
    # input()
    
    if net_mode == 'student':
        for name, param in model.named_parameters():
            if name in trainable_names:
                param.requires_grad = True
                print_ddp(byello(f'[Student] Set true for: {name}'))
            else:
                param.requires_grad = False
                print_ddp(gray(f'[Student] Set false for: {name}'))
    else:
        for name, param in model.named_parameters():
            param.requires_grad = False
            print_ddp(gray(f'[Teacher] Set false for: {name}'))
        
    # print_ddp(f'Set requires_grad to FALSE for all the params')
    print_ddp(cyan(f'Successfully loaded weights into APLA model with mode: {mode}'))
    return model

