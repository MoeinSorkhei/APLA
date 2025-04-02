import torch
from functools import partial
from torch.nn.parallel import DistributedDataParallel as DDP
import inspect
import os

from defaults import DefaultWrapper
from defaults.datasets import edict
from defaults.models import edict
from self_supervised.DINO import DINOWrapper
from utils._utils import edict
from utils.dist_utills import ddp_is_on
from utils import helpfuns

from .dinov2_utils import collate_data_and_cast
from .dinov2_utils import MaskingGenerator
from .models import DINOv2


class DINOv2Wrapper(DINOWrapper):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.params = parameters
        self.set_crops_params()  # set some infor about crops from the strategy file
        self.train_collate_fn = self.get_train_collate_fn()
        self.schedulers = None  # handled separately during training
    
    @property
    def transformers_params(self):
        return self.params.model_params.transformers_params
    
    @property 
    def ibot_params(self):
        return self.params.model_params.dinov2.ibot
    
    def set_crops_params(self):
        self_dir = os.path.dirname(os.path.abspath(inspect.getfile(self.__class__)))
        strategy_path = os.path.join(self_dir, "augmentation_strategy.json")
        strategy = edict(helpfuns.load_json(strategy_path))
        
        assert len(strategy.repetition_strategy.n_augmentations) == 3  # ["global_1", "global_2", "local"]
        train_transforms = strategy.transforms.train_transforms
        self.params.crops_params = edict(
            n_global_crops=sum(strategy.repetition_strategy.n_augmentations[:2]),
            n_local_crops=strategy.repetition_strategy.n_augmentations[2],
            global_crops_size=train_transforms.global_1.RandomResizedCrop.size,
            local_crops_size=train_transforms.local.RandomResizedCrop.size,
            img_size=train_transforms.global_1.RandomResizedCrop.size  # same as global
        )
    
    def get_train_collate_fn(self):
        img_size = self.params.crops_params.img_size
        patch_size = self.transformers_params.student.patch_size
        inputs_dtype = torch.float32  # torch.half
        n_tokens = (img_size // patch_size) ** 2
        
        mask_generator = MaskingGenerator(
            input_size=(img_size // patch_size, img_size // patch_size),
            max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
        )
        collate_fn = partial(
            collate_data_and_cast,
            n_global_crops=self.params.crops_params.n_global_crops,
            n_local_crops=self.params.crops_params.n_local_crops,
            mask_ratio_tuple=self.ibot_params.mask_ratio_min_max,
            mask_probability=self.ibot_params.mask_sample_probability,
            n_tokens=n_tokens,
            mask_generator=mask_generator,
            dtype=inputs_dtype,
        )
        return collate_fn
    
    def init_model(self):
        model = DINOv2(params=self.params)
        model.to(self.device_id)
        if ddp_is_on():
            model = DDP(model, device_ids=[self.device_id])
        return model
    
    def init_criteria(self):          
        # define criteria
        crit = None
        return crit
