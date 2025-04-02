# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import random
import numpy as np
import random
import math
import numpy as np
from torch import nn

"""
collate_fn gets a list of items, where each item is the output of dataset __getitem__
it processes the items and converts crops, labels, etc. them to torch tensors (still on cpu)
it finally returns a dictionary that contains the batchified data.
later in the model forward() the dictionary is parsed and the tensors are moved to gpu
"""

def collate_data_and_cast(samples_list, n_global_crops, n_local_crops, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None):
    # samples_list is the output of __getitem__, containing a list of globa + local crops + label
    collated_global_crops = torch.stack([s[0][i] for i in range(n_global_crops) for s in samples_list])
    collated_local_crops = torch.stack([s[0][i] for i in range(n_global_crops, n_global_crops + n_local_crops) for s in samples_list])
    labels = torch.cat([s[1].unsqueeze(0) for s in samples_list], dim=0)
    # shapes
    # global crops: torch.Size([B * n_global_crops, 3, 224, 224])
    # local crops: torch.Size([B * n_local_crops, 3, 98, 98])

    B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]
    # still on cpu
    return {
        "images": {
            "collated_global_crops": collated_global_crops.to(dtype),
            "collated_local_crops": collated_local_crops.to(dtype),
            "collated_masks": collated_masks,
            "mask_indices_list": mask_indices_list,
            "masks_weight": masks_weight,
            "upperbound": upperbound,
            "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
        },
        "labels": labels,
    }


class MaskingGenerator:
    def __init__(
        self,
        input_size,
        num_masking_patches=None,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    # def __repr__(self):
    #     repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
    #         self.height,
    #         self.width,
    #         self.min_num_patches,
    #         self.max_num_patches,
    #         self.num_masking_patches,
    #         self.log_aspect_ratio[0],
    #         self.log_aspect_ratio[1],
    #     )
    #     return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches=0):
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask


class CosineScheduler(object):
    def __init__(self, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, freeze_iters=0):
        super().__init__()
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros((freeze_iters))

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))

        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False

