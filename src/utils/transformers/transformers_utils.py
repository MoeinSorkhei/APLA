import math
import torch
import warnings
from utils.dist_utills import print_ddp
from utils.colors import *
from utils import helpfuns
import timm


def download_weights(backbone_type, patch_size, pretrained_type):
    supported_pretrained = ['dinov2']
    assert pretrained_type in supported_pretrained, f'pretrained_type should be in {supported_pretrained}'
    
    checkpoints_dinov2 = {  # all without register
        "vit_small": {
            14: "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
        },
        "vit_base": {
            14: "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
        },
        "vit_large": {
            14: "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth"
        },
        "vit_giant": {
            14: "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth"
        }
    }
    checkpoints = {
        "dinov2": checkpoints_dinov2,
    }
    
    try:
        # Set the SSL certificate file for secure download
        # helpfuns.install('certifi')
        import certifi
        import os
        os.environ['SSL_CERT_FILE'] = certifi.where()
        #
        url = checkpoints[pretrained_type][backbone_type][patch_size]
        print_ddp(green(f'Using {backbone_type} {pretrained_type} initialization from: {url}'))
        chpt = torch.hub.load_state_dict_from_url(url, progress=True)
    except:
        raise ValueError(f"Pretrained weights for {backbone_type} with patch size {patch_size} with pretrained method {pretrained_type} not found.")
        
    if pretrained_type == 'dinov2':
        extra_keys = extra_keys = ['mask_token']  # , 'ls1', 'ls2']   # to be consistent with original deit def
        chpt = {key: value for key, value in chpt.items() if not any(sub in key for sub in extra_keys)}
    
    if pretrained_type in ['in21k', 'in21k_5']:
        if 'swin' in backbone_type:
            chpt = chpt['model']
        else:
            chpt = chpt['state_dict']
        del chpt['head.weight']
        del chpt['head.bias']
    
    return chpt
