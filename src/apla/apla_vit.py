import torch
import torch.nn as nn
from utils.transformers import vit
from utils.transformers.transformers_utils import download_weights
import torch.nn.functional as F
import warnings

from utils import *
from utils.colors import *
from utils import helpfuns


class APLA_Attention(nn.Module):
    def __init__(self, config, dim, indices=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.partial_size = config.partial_size
        self.dim = dim

        # if indices is provided, use them (using pre-defined inds)
        # otherwise sample indices ONCE at initialization
        if indices is not None:
            self.indices = indices
            print_ddp(f'APLA_Attention init: Using provided indices')
        else:
            self.indices = torch.randperm(self.dim)
            print_ddp(f'APLA_Attention init: Sampled a set of random indices')
        
        # register the indices as buffer
        self.register_buffer('inds', self.indices)
        
        # define trainable/frozen inds
        self.trainable_inds = self.indices[:self.partial_size]
        self.freezed_inds = self.indices[self.partial_size:]

        # qkv projection, kepts frozen
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        for param in self.qkv.parameters():
            param.requires_grad = False

        # output projection
        self.proj_weight1 = nn.Parameter(torch.empty(self.partial_size, dim), requires_grad=True)
        self.proj_weight2 = nn.Parameter(torch.empty(dim - self.partial_size, dim), requires_grad=False)
        self.proj_bias1 = nn.Parameter(torch.empty(self.partial_size), requires_grad=True)
        self.proj_bias2 = nn.Parameter(torch.empty(dim - self.partial_size), requires_grad=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # qkv projection, this is exactly the same as original ViT
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # output projection
        # forward passes with trainable and freezed params
        trainable_out = F.linear(x, self.proj_weight1, self.proj_bias1)
        freezed_out = F.linear(x, self.proj_weight2, self.proj_bias2)
        output = torch.empty(x.shape, device=x.device, dtype=trainable_out.dtype)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # Ignore scatter_ tensor warning
            output.scatter_(  # trainable part
                dim=-1,
                index=torch.tensor(self.trainable_inds, device=x.device).unsqueeze(0).unsqueeze(0).expand(x.size(0), x.size(1), -1),  # noqa
                src=trainable_out
            )
            output.scatter_(  # freezed part
                dim=-1,
                index=torch.tensor(self.freezed_inds, device=x.device).unsqueeze(0).unsqueeze(0).expand(x.size(0), x.size(1), -1),  # noqa
                src=freezed_out
            )

        x = output
        x = self.proj_drop(x)
        return x, attn


def replace_attn_with_apla(model, config):
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
        apla_attn = APLA_Attention(
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
        print_ddp(f'Replaced Attention in block {i} with APLA_Attention')


def build_apla(config, backbone_type, transformers_params, pretrained, is_multi_gpu=False):
    model = vit.__dict__[backbone_type](**transformers_params, pretrained=pretrained)
    
    if is_multi_gpu:
        if config.partial_size == 'full':
            for name, param in model.named_parameters():
                if 'attn.proj' in name:
                    param.requires_grad = True
                    print_ddp(byello(f'Set true for: {name}'))
                else:
                    param.requires_grad = False
                    print_ddp(gray(f'Set false for: {name}'))
            print_ddp(cyan(f'Successfully built APLA-enabled model\n'))
            return model
        else:
            assert 'inds_path' in config, '"inds_path" should be present with multi-gpu training with random sampling'
    
    # freezing all params
    for _, param in model.named_parameters():
        param.requires_grad = False
    
    # replace attention with apla attention (contains trainable parameters)
    replace_attn_with_apla(model=model, config=config)
    print_ddp(f'Replaced Attention modules with Attention_APLA')
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print_ddp(byello(f'{name} requires_grad: {param.requires_grad}'))
        else:
            print_ddp(gray(f'{name} requires_grad: {param.requires_grad}'))
        
    print_ddp(cyan(f'Successfully built APLA-enabled model\n'))
    return model
