import torch

from utils import *
from utils.colors import *
from utils import helpfuns

from .appla_attn import APLA_Attention
from .appla_attn_mem_eff import APLA_MemEffAttention


def replace_attn_with_apla(model, config, attn_module):
    if hasattr(config, 'inds_path'):
        print_ddp(cyan(f'Registering inds based on path: {config.inds_path}'))
        
    for i, block in enumerate(model.blocks):
        # access the attention module of the block
        attn = block.attn

        # if pre-defined trainable indices exist, use them
        if hasattr(config, 'inds_path'):
            inds_dict = helpfuns.load_json(config.inds_path)
            trainable_inds = inds_dict[f'block_{i}']  # contains trainable inds
            freezed_inds = [idx for idx in range(attn.dim) if idx not in trainable_inds]
            indices = torch.tensor(trainable_inds + freezed_inds)
        else:
            indices = None

        # initialize apla attention and sample trainable matrix rows
        apla_attn = attn_module(
            config=config, 
            dim=attn.dim,
            indices=indices,
            num_heads=attn.num_heads,
            qkv_bias=attn.qkv.bias is not None,
            qk_scale=attn.scale,
            attn_drop=attn.attn_drop.p,
            proj_drop=attn.proj_drop.p
        )

        # === copy pretrained weights ===
        with torch.no_grad():
            # copy the qkv weights
            apla_attn.qkv.weight.data = attn.qkv.weight.data.clone()
            if attn.qkv.bias is not None:
                apla_attn.qkv.bias.data = attn.qkv.bias.data.clone()

            # copy the projection weights (split into trainable & frozen parts)
            full_proj_weight = attn.proj.weight.data.clone()
            full_proj_bias = attn.proj.bias.data.clone() if attn.proj.bias is not None else None

            apla_attn.proj_weight1.data = full_proj_weight[apla_attn.trainable_inds, :]
            apla_attn.proj_weight2.data = full_proj_weight[apla_attn.freezed_inds, :]

            if full_proj_bias is not None:
                apla_attn.proj_bias1.data = full_proj_bias[apla_attn.trainable_inds]
                apla_attn.proj_bias2.data = full_proj_bias[apla_attn.freezed_inds]

        # replace the original attention module with apla attention
        block.attn = apla_attn
        print_ddp(f'Replaced {attn.__class__.__name__} in block {i} with {apla_attn.__class__.__name__}')


def build_apla(config, model, attn_class, is_multi_gpu=False):
    # model = vit.__dict__[backbone_type](**transformers_params, pretrained=pretrained)
    if is_multi_gpu:
        if config.partial_size == 'full':
            for name, param in model.named_parameters():
                if 'attn.proj' in name:
                    param.requires_grad = True
                    print_ddp(byello(f'Building apla -- Set requires_grad to true for: {name}'))
                else:
                    param.requires_grad = False
                    print_ddp(gray(f'Building apla -- Set requires_grad to false for: {name}'))
            print_ddp(cyan(f'Successfully built APLA-enabled model\n'))
            return model
        else:
            assert 'inds_path' in config, '"inds_path" should be present with multi-gpu training with random sampling'
    
    # freezing all params
    for _, param in model.named_parameters():
        param.requires_grad = False
    
    # replace attention with apla attention (contains trainable parameters)
    if attn_class == 'apla_attn':
        attn_module = APLA_Attention
    elif attn_class == 'apla_attn_mem_eff':
        attn_module = APLA_MemEffAttention
    else:
        raise NotImplementedError
    
    replace_attn_with_apla(model=model, config=config, attn_module=attn_module)
    print_ddp(f'Replaced Attention modules with Attention_APLA')
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print_ddp(byello(f'Building apla -- {name} requires_grad: {param.requires_grad}'))
        else:
            print_ddp(gray(f'Building apla -- {name} requires_grad: {param.requires_grad}'))
        
    print_ddp(cyan(f'Successfully built APLA-enabled model\n'))
    return model
