import os
import torch
import pprint

from utils.colors import *


def state_dict_from_path(path):
    """
    Returns the state dict from a checkpoint path
    """
    path = os.path.abspath(path)
    mname = os.path.basename(path)
    if os.path.isfile(path):
        # print("Loading weights from \"{}\"".format(mname))
        return torch.load(path)['state_dict']
    else:
        dirname = os.path.dirname(path)
        raise FileNotFoundError(
            "Model \"{}\" is not present in \"{}\"".format(mname, dirname))


def load_from_pretrained(model, path, strict=False):
    """
    Loads wights to the input model from a pretrained model path
    """
    if 'fastadapt' in path or 'apla' in path:
        missing_keys, unexpected_keys = model.load_state_dict(torch.load(path)['state_dict'], strict=False)
        assert missing_keys == [], f'There are unexpected keys!: \n{unexpected_keys}\n'
        assert all(['partial_size' in key for key in unexpected_keys])
    else:
        pretrained_state = state_dict_from_path(path)
        # dif_keys = model.backbone.load_state_dict(pretrained_state, strict=strict)
        dif_keys = model.load_state_dict(pretrained_state, strict=True)
        dif_keys = set([" : ".join(key.split(".")[:2]) for key in dif_keys.unexpected_keys])
        if dif_keys:
            print("Unmatched pretrained modules")
            pprint.pprint(dif_keys)
    print_byello(f'+++++ Successfully loaded weights from: {path}')
