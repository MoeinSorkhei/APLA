import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from utils import *
from utils.colors import *


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

