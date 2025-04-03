import torch.nn as nn

from mmengine.logging import print_log  # type: ignore
from mmseg.registry import MODELS  # type: ignore
from .vit import VisionTransformer  # type: ignore


def count_trainable_params(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return count


def count_total_params(model):
    count = sum(p.numel() for p in model.parameters())
    return count


@MODELS.register_module()
class APLA_VisionTransformer(VisionTransformer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # print(f'img_size: {self.img_size}')
        # input()
        
        for name, param in self.named_parameters():
            if 'attn.out_proj' in name:
                param.requires_grad = True
                print_log(f'Set requires_grad to TRUE for {name}')
            else:
                param.requires_grad = False
                print_log(f'Set requires_grad to FALSE for {name}')
        
        print_log(f'\nTotal params: {count_total_params(self):,}')
        print_log(f'Trainable params: {count_trainable_params(self):,}\n')
        # input()

    # def forward(self, x):  # should return a tuple
    #     pass

    # def init_weights(self, pretrained=None):
    #     pass
    

