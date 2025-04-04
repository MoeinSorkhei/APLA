from torch.cuda.amp import autocast

from .bases import *
from utils.transformers import *
from utils.colors import *
from utils import *
from utils.transformers import vit


class Identity(nn.Module):
    """An identity function."""
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
    
class Classifier(BaseModel):
    """
    A wrapper class that provides different CNN backbones.
    Is not intended to be used standalone. Called using the DefaultWrapper class.
    """
    def __init__(self, model_params, system_params):
        super().__init__()
        self.attr_from_dict(model_params)
        
        # adaptation enabled
        if 'adaptation' in model_params:
            print_ddp(blue((f'\nADAPTATION ENABLED with: {model_params.adaptation.mode}!!')))
            mode = model_params.adaptation.mode
            adaptation_params = model_params.adaptation.params
            transformers_params = model_params.transformers_params
            assert mode == 'apla', 'Only adaptation with APLA is enabled'
            assert 'vit' in self.backbone_type, 'Only supports ViT with multi-gpu training'
            
            if 'vit' in self.backbone_type:
                from apla import apla_vit as apla_model
                model = vit.__dict__[self.backbone_type](**transformers_params, pretrained=model_params.pretrained)
            
            elif 'swin' in self.backbone_type:
                raise NotImplementedError
                # from apla import apla_swin as apla_model
            
            else:
                raise ValueError
            
            # build apla
            self.backbone = apla_model.build_apla(
                config=adaptation_params,
                model=model,
                attn_class='apla_attn',
                is_multi_gpu=len(system_params.which_GPUs.split(",")) > 1,
            )
        
        # --- init transformers
        elif hasattr(transformers, self.backbone_type):
            print_byello(f'No Adaptation...')
            self.backbone = transformers.__dict__[self.backbone_type](**self.transformers_params, pretrained=self.pretrained)
        
        else:
            raise NotImplementedError                
        
        self.backbone.fc = Identity()  # removing the fc layer from the backbone (which is manually added below)
        self.fc = nn.Linear(self.backbone.num_features, self.n_classes)
        
        if self.freeze_backbone:
            self.freeze_submodel(self.backbone)
        
        print_ddp(blue(
            f'Model total params (inc. FC)   : {helpfuns.count_total_params(self):,} '
            f'-- model trainable params (inc. FC): {helpfuns.count_trainable_params(self):,}\n'
            f'Backbone total params (exc. FC): {helpfuns.count_total_params(self.backbone):,} '
            f'-- backbone trainable params (exc. FC): {helpfuns.count_trainable_params(self.backbone):,}\n'
        ))

    @property
    def pretrained_type(self):
        return self.transformers_params.pretrained_type
    
    def forward(self, x, return_embedding=False):
        with autocast(self.use_mixed_precision):
            if self.freeze_backbone:
                self.backbone.eval()
            
            x_emb = self.backbone(x)
            x = self.fc(x_emb)
            
            if return_embedding:
                return x, x_emb        
            else:
                return x
        
    # def modify_first_layer(self, img_channels, pretrained):
    #     backbone_type = self.backbone.__class__.__name__
    #     if img_channels == 3:
    #         return

    #     if backbone_type == 'ResNet':
    #         conv_attrs = ['out_channels', 'kernel_size', 'stride', 
    #                       'padding', 'dilation', "groups", "bias", "padding_mode"]
    #         conv1_defs = {attr: getattr(self.backbone.conv1, attr) for attr in conv_attrs}

    #         pretrained_weight = self.backbone.conv1.weight.data
    #         pretrained_weight = pretrained_weight.repeat(1, 4, 1, 1)[:, :img_channels]

    #         self.backbone.conv1 = nn.Conv2d(img_channels, **conv1_defs)
    #         if pretrained:
    #             self.backbone.conv1.weight.data = pretrained_weight 
                
    #     elif backbone_type == 'Inception3':
    #         conv_attrs = ['out_channels', 'kernel_size', 'stride', 
    #                       'padding', 'dilation', "groups", "bias", "padding_mode"]
    #         conv1_defs = {attr: getattr(self.backbone.Conv2d_1a_3x3.conv, attr) for attr in conv_attrs}

    #         pretrained_weight = self.backbone.Conv2d_1a_3x3.conv.weight.data
    #         pretrained_weight = pretrained_weight.repeat(1, 4, 1, 1)[:, :img_channels]

    #         self.backbone.Conv2d_1a_3x3.conv = nn.Conv2d(img_channels, **conv1_defs)
    #         if pretrained:
    #             self.backbone.Conv2d_1a_3x3.conv.weight.data = pretrained_weight                 
                
    #     elif backbone_type == 'VisionTransformer':
    #         patch_embed_attrs = ["img_size", "patch_size", "embed_dim"]
    #         patch_defs = {attr: getattr(self.backbone.patch_embed, attr) for attr in patch_embed_attrs}

    #         pretrained_weight = self.backbone.patch_embed.proj.weight.data
    #         if self.backbone.patch_embed.proj.bias is not None:
    #             pretrained_bias = self.backbone.patch_embed.proj.bias.data
    #         pretrained_weight = pretrained_weight.repeat(1, 4, 1, 1)[:, :img_channels]
            
    #         self.backbone.patch_embed = transformers.vit.PatchEmbed(in_chans=img_channels, **patch_defs)
    #         if pretrained:
    #             self.backbone.patch_embed.proj.weight.data = pretrained_weight 
    #             if self.backbone.patch_embed.proj.bias is not None:
    #                 self.backbone.patch_embed.proj.bias.data = pretrained_bias           
                    
    #     elif backbone_type == 'SwinTransformer':
    #         patch_embed_attrs = ["img_size", "patch_size", "embed_dim", "norm_layer"]
    #         patch_defs = {attr: getattr(self.backbone.patch_embed, attr) for attr in patch_embed_attrs}

    #         pretrained_weight = self.backbone.patch_embed.proj.weight.data
    #         if self.backbone.patch_embed.proj.bias is not None:
    #             pretrained_bias = self.backbone.patch_embed.proj.bias.data
    #         if self.backbone.patch_embed.norm is not None:
    #             norm_weight = self.backbone.patch_embed.norm.weight.data                
    #             norm_bias = self.backbone.patch_embed.norm.bias.data                
    #         pretrained_weight = pretrained_weight.repeat(1, 4, 1, 1)[:, :img_channels]
            
    #         self.backbone.patch_embed = transformers.swin.PatchEmbed(in_chans=img_channels, **patch_defs)
    #         if pretrained:
    #             self.backbone.patch_embed.proj.weight.data = pretrained_weight 
    #             if self.backbone.patch_embed.proj.bias is not None:
    #                 self.backbone.patch_embed.proj.bias.data = pretrained_bias      
    #             if self.backbone.patch_embed.norm is not None:
    #                 if self.backbone.patch_embed.norm.weight is not None:
    #                     self.backbone.patch_embed.norm.weight.data = norm_weight
    #                 if self.backbone.patch_embed.norm.bias is not None:
    #                     self.backbone.patch_embed.norm.bias.data = norm_bias
                        
    #     elif backbone_type == 'FocalTransformer':
    #         raise NotImplementedError
    #         patch_embed_attrs = ["img_size", "patch_size", "embed_dim", "norm_layer",
    #                              "use_conv_embed", "norm_layer", "use_pre_norm", "is_stem"]
    #         patch_defs = {attr: getattr(self.backbone.patch_embed, attr) for attr in patch_embed_attrs}

    #         pretrained_weight = self.backbone.patch_embed.proj.weight.data
    #         if self.backbone.patch_embed.proj.bias is not None:
    #             pretrained_bias = self.backbone.patch_embed.proj.bias.data
    #         if self.backbone.patch_embed.norm is not None:
    #             norm_weight = self.backbone.patch_embed.norm.weight.data                
    #             norm_bias = self.backbone.patch_embed.norm.bias.data 
    #         if self.backbone.patch_embed.pre_norm is not None:
    #             norm_weight = self.backbone.patch_embed.pre_norm.weight.data                
    #             norm_bias = self.backbone.patch_embed.pre_norm.bias.data                 
    #         pretrained_weight = pretrained_weight.repeat(1, 4, 1, 1)[:, :img_channels]
            
    #         self.backbone.patch_embed = transformers.focal.PatchEmbed(in_chans=img_channels, **patch_defs)
    #         if pretrained:
    #             self.backbone.patch_embed.proj.weight.data = pretrained_weight 
    #             if self.backbone.patch_embed.proj.bias is not None:
    #                 self.backbone.patch_embed.proj.bias.data = pretrained_bias      
    #             if self.backbone.patch_embed.norm is not None:
    #                 if self.backbone.patch_embed.norm.weight is not None:
    #                     self.backbone.patch_embed.norm.weight.data = norm_weight
    #                 if self.backbone.patch_embed.norm.bias is not None:
    #                     self.backbone.patch_embed.norm.bias.data = norm_bias 
    #                 if self.backbone.patch_embed.pre_norm.weight is not None:
    #                     self.backbone.patch_embed.pre_norm.weight.data = pre_norm_weight
    #                 if self.backbone.patch_embed.pre_norm.bias is not None:
    #                     self.backbone.patch_embed.pre_norm.bias.data = pre_norm_bias                         
        
    #     else:
    #         raise NotImplementedError("channel modification is not implemented for {}".format(backbone_type))
