import os
import warnings
from torch import Tensor
import torch
import torch.nn.functional as F

from .appla_attn import APLA_Attention


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None

try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind
        XFORMERS_AVAILABLE = True
    else:
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False


class APLA_MemEffAttention(APLA_Attention):
    """
    This is very similar to APLA_Attention, except that is uses memory-efficient attetion
    as done in Dinov2.
    """
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        # MemEff attention from dinov2
        # References:
        #   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
        #   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        # APLA on output projection
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
        return x
