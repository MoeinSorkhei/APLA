import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mmengine.logging import print_log  # type: ignore

from .swin_transformer import SwinTransformer  # type: ignore
from mmcv_custom import load_checkpoint  # type: ignore
from mmdet.utils import get_root_logger  # type: ignore
from ..builder import BACKBONES  # type: ignore


def count_trainable_params(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return count


def count_total_params(model):
    count = sum(p.numel() for p in model.parameters())
    return count


@BACKBONES.register_module()
class APLA_SwinTransformer(SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        for name, param in self.named_parameters():
            if 'attn.proj' in name:
                param.requires_grad = True
                print_log(f'Set requires_grad to TRUE for {name}')
            else:
                param.requires_grad = False
                print_log(f'Set requires_grad to FALSE for {name}')
        
        print_log(f'\nTotal params: {count_total_params(self):,}')
        print_log(f'Trainable params: {count_trainable_params(self):,}\n')

    