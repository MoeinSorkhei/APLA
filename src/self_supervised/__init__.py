from .BYOL import *
from .DINO import *
# from .dinov2 import *  # do not uncomment this -- will mess up with setting gpu correctly

__all__ = [k for k in globals().keys() if not k.startswith("_")]