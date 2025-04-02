# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import os
import warnings

from torch import Tensor
from torch import nn
import torch
import random
import torch.nn.functional as F


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None

try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        # warnings.warn("xFormers is available (Attention)")
    else:
        # warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    # warnings.warn("xFormers is not available (Attention)")



def attn_has(mode, element):
    if mode == 'attn' or element in mode:
        return True
    return False


def mlp_has(mode, element):
    if mode == 'mlp' or element in mode:
        return True
    return False


class Attention(nn.Module):
    def __init__(
        self,
        conf,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        
        partial_size = conf.partial_size
        self.partial_size = conf.partial_size
        self.mode = conf.mode
        self.ind_mode = conf.ind_mode
        self.dim = dim

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        if attn_has(self.mode, 'proj'):
            # self.register_buffer('indices', torch.arange(self.dim))
            # self.register_buffer('indices', torch.randperm(self.dim))
            self.proj_weight1 = nn.Parameter(torch.empty(self.partial_size, dim), requires_grad=True)
            self.proj_weight2 = nn.Parameter(torch.empty(dim - self.partial_size, dim), requires_grad=False)
            self.proj_bias1 = nn.Parameter(torch.empty(self.partial_size), requires_grad=True)
            self.proj_bias2 = nn.Parameter(torch.empty(dim - self.partial_size), requires_grad=False)
        else:
            self.proj = nn.Linear(dim, dim)
        # self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # x = self.proj(x)
        x = self.proj_drop(x)
        return x


class APLA_MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        if attn_has(self.mode, 'proj'):
            # with torch.no_grad():
            #     trainable_indices = self.indices[:self.partial_size]
            #     freezed_indices = self.indices[self.partial_size:]
            #     # concat with correct indices
            #     proj_weight = torch.empty(self.dim, self.dim, device=x.device)
            #     proj_weight[trainable_indices] = self.proj_weight1
            #     proj_weight[freezed_indices] = self.proj_weight2
            #     #
            #     proj_bias = torch.empty(self.dim, device=x.device)
            #     proj_bias[trainable_indices] = self.proj_bias1
            #     proj_bias[freezed_indices] = self.proj_bias2
                
            # self.proj_weight1.data = proj_weight[trainable_indices]
            # self.proj_weight2.data = proj_weight[freezed_indices]
            # self.proj_bias1.data = proj_bias[trainable_indices]
            # self.proj_bias2.data = proj_bias[freezed_indices]
            # individual linear
            trainable_out = F.linear(x, self.proj_weight1, self.proj_bias1)
            freezed_out = F.linear(x, self.proj_weight2, self.proj_bias2)
            output = torch.empty(x.shape, device=x.device, dtype=trainable_out.dtype)
            #
            trainable_indices = self.indices[:self.partial_size]
            freezed_indices = self.indices[self.partial_size:]
            #
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)  # Could not figure it out and has no effect for torch.tensor warning below
                output.scatter_(  # trainable part
                    dim=-1, 
                    index=torch.tensor(trainable_indices, device=x.device).unsqueeze(0).unsqueeze(0).expand(x.size(0), x.size(1), -1), # noqa
                    src=trainable_out
                )
                output.scatter_(  # freezed part
                    dim=-1, 
                    index=torch.tensor(freezed_indices, device=x.device).unsqueeze(0).unsqueeze(0).expand(x.size(0), x.size(1), -1), # noqa
                    src=freezed_out
                )
            x = output
            # prepare for the next iteration
            with torch.no_grad():
                # concat with correct indices
                proj_weight = torch.empty(self.dim, self.dim, device=x.device)
                proj_weight[trainable_indices] = self.proj_weight1
                proj_weight[freezed_indices] = self.proj_weight2
                #
                proj_bias = torch.empty(self.dim, device=x.device)
                proj_bias[trainable_indices] = self.proj_bias1
                proj_bias[freezed_indices] = self.proj_bias2
                
                if self.ind_mode == 'v3':
                    self.indices = torch.randperm(self.dim)
                    # print(f'For ind_mode: {self.ind_mode} -- resmapled inds -- now inds: {self.indices[:10]}')
                    # input()
                if self.ind_mode == 'v5':
                    self.indices = torch.randperm(self.dim)
                    self.partial_size = random.randint(1, self.dim)
                    # print(f'For ind_mode: {self.ind_mode} -- resmapled inds -- now inds: {self.indices[:10]}')
                    # print(f'For ind_mode: {self.ind_mode} -- Also resmapled partial size: {self.partial_size}')
                    self.proj_weight1 = nn.Parameter(torch.empty(self.partial_size, self.dim), requires_grad=True)
                    self.proj_weight2 = nn.Parameter(torch.empty(self.dim - self.partial_size, self.dim), requires_grad=False)
                    self.proj_bias1 = nn.Parameter(torch.empty(self.partial_size), requires_grad=True)
                    self.proj_bias2 = nn.Parameter(torch.empty(self.dim - self.partial_size), requires_grad=False)
                    # input()
                trainable_indices = self.indices[:self.partial_size]
                freezed_indices = self.indices[self.partial_size:]
                
            self.proj_weight1.data = proj_weight[trainable_indices]
            self.proj_weight2.data = proj_weight[freezed_indices]
            self.proj_bias1.data = proj_bias[trainable_indices]
            self.proj_bias2.data = proj_bias[freezed_indices]
            
        else:
            x = self.proj(x)
        # x = self.proj(x)
        x = self.proj_drop(x)
        return x
