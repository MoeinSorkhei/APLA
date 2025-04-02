from torch import nn
from torchvision.transforms import *
from torch.utils.data import Dataset
import wandb
import pprint as the_pprint

from utils import *
from utils.colors import *


DEFAULT_INTERPOLATION_MODE = InterpolationMode.BICUBIC


class BaseSet(Dataset):
    """Base dataset class that actual datasets, e.g. Cifar10, subclasses.

    This class only has torchvision.transforms for augmentation.
    Not intended to be used directly.
    """
    def __init__(self):
        super().__init__()  
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        if 'img_path' in self.data[idx]:
            img_path = self.data[idx]['img_path']
            png_path = '.'.join(img_path.split('.')[:-1]) + '.png'
            if os.path.exists(png_path):
                img = self.get_x(png_path)
                img_path = png_path
            else:
                img = self.get_x(img_path)
        
        elif 'img_arr' in self.data[idx]:   # array should be 3 channels for RGB -- for CIFAR
                img_arr = self.data[idx]['img_arr']
                img = Image.fromarray(img_arr)
        
        else:
            raise NotImplementedError
        
        label = torch.as_tensor(self.data[idx]['label'])

        if self.resizing is not None:
            img = self.resizing(img)

        if self.transform is not None:
            if isinstance(self.transform, list):
                # SSL training, e.g. dino, dinov2 etc. (augmentation_strategy file exists)
                img = [tr(img) for tr in self.transform]
            else:
                # normal supervised training with one 1 augmentation
                img = [self.transform(img) for _ in range(self.num_augmentations)]
            img = img[0] if len(img) == 1 and isinstance(img, list) else img      
            
        # for dino etc img is a list of tensors (crops), for simple fineutning it's a tensor
        return img, label
            
    def get_x(self, img_path):
        return pil_loader(img_path, self.img_channels)    
    
    def attr_from_dict(self, param_dict):
        self.num_augmentations = 1        
        self.name = self.__class__.__name__
        for key in param_dict:
            setattr(self, key, param_dict[key])

    def get_trans_list(self, transform_dict):
        transform_list = []   
        # ----------------------------------------
        # first: primarly transformations 
        # geometric augmentation (resizing, scaling, translating, flipping, etc.)
        # ----------------------------------------
        if "Resize" in transform_dict:
            if transform_dict['Resize']['apply']:
                transform_list.append(
                    Resize(
                        size=(transform_dict['Resize']['height'], transform_dict['Resize']['width']),
                        interpolation=DEFAULT_INTERPOLATION_MODE
                    )
                )

        if "CenterCrop" in transform_dict:                
            if transform_dict['CenterCrop']['apply']:
                transform_list.append(CenterCrop((transform_dict['CenterCrop']['height'],
                                                 transform_dict['CenterCrop']['width'])))   

        if "RandomCrop" in transform_dict:                
            if transform_dict['RandomCrop']['apply']:
                padding = transform_dict['RandomCrop']['padding']
                transform_list.append(RandomCrop((transform_dict['RandomCrop']['height'],
                                             transform_dict['RandomCrop']['width']),
                                            padding=padding if padding > 0 else None))            

        if "RandomResizedCrop" in transform_dict:                
            if transform_dict['RandomResizedCrop']['apply']:
                transform_list.append(RandomResizedCrop(size=transform_dict['RandomResizedCrop']['size'],
                                                       scale=transform_dict['RandomResizedCrop']['scale'],
                                                       interpolation=DEFAULT_INTERPOLATION_MODE))        

        if "VerticalFlip" in transform_dict:                
            if transform_dict['VerticalFlip']['apply']:
                transform_list.append(RandomVerticalFlip(p=transform_dict['VerticalFlip']['p']))

        if "HorizontalFlip" in transform_dict:                
            if transform_dict['HorizontalFlip']['apply']:
                transform_list.append(RandomHorizontalFlip(p=transform_dict['HorizontalFlip']['p']))

        if "RandomRotation" in transform_dict:                
            if transform_dict['RandomRotation']['apply']:
                transform_list.append(
                    rand_apply(RandomRotation(degrees=transform_dict['RandomRotation']['angle']),
                                                      p=transform_dict['RandomRotation']['p']))

        # ----------------------------------------
        # second: pixel-based augmentations (color jitter etc.)
        # ----------------------------------------
        if "ColorJitter" in transform_dict:                
            if transform_dict['ColorJitter']['apply']:
                temp_d = transform_dict['ColorJitter']
                transform_list.append(
                    rand_apply(ColorJitter(brightness=temp_d['brightness'],
                                                  contrast=temp_d['contrast'], 
                                                  saturation=temp_d['saturation'], 
                                                  hue=temp_d['hue']),
                                                  p=temp_d['p'])) 

        if "RandomGrayscale" in transform_dict:                
            if transform_dict['RandomGrayscale']['apply']:
                transform_list.append(RandomGrayscale(p=transform_dict['RandomGrayscale']['p']))             

        if "RandomGaussianBlur" in transform_dict:                
            if transform_dict['RandomGaussianBlur']['apply']:
                transform_list.append(
                    RandomGaussianBlur(p=transform_dict['RandomGaussianBlur']['p'],
                                       radius_min=transform_dict['RandomGaussianBlur']['radius_min'],
                                       radius_max=transform_dict['RandomGaussianBlur']['radius_max'])) 

        if "RandomAffine" in transform_dict:                
            if transform_dict['RandomAffine']['apply']:
                temp_d = transform_dict['RandomAffine']
                transform_list.append(
                    rand_apply(RandomAffine(degrees=temp_d['degrees'],
                                                  translate=temp_d['translate'], 
                                                  scale=temp_d['scale'], 
                                                  shear=temp_d['shear']),
                                                  p=temp_d['p']))                    

        if "RandomPerspective" in transform_dict:                
            if transform_dict['RandomPerspective']['apply']:
                transform_list.append(RandomPerspective(transform_dict['RandomPerspective']['distortion_scale'],
                                  p=transform_dict['RandomPerspective']['p']))  

        if "RandomSolarize" in transform_dict:                
            if transform_dict['RandomSolarize']['apply']:
                transform_list.append(RandomSolarize(threshold=transform_dict['RandomSolarize']['threshold'],
                                                     p=transform_dict['RandomSolarize']['p']))              

        if "AugMix" in transform_dict:
            if transform_dict['AugMix']['apply']:
                from utils.augmix import AugMix
                transform_list.append(
                    AugMix(
                        severity=transform_dict['AugMix']['severity'],
                        mixture_width=transform_dict['AugMix']['mixture_width'],
                        chain_depth=transform_dict['AugMix']['chain_depth'],
                        alpha=transform_dict['AugMix']['alpha'],
                        all_ops=transform_dict['AugMix']['all_ops'],
                        interpolation=InterpolationMode.BILINEAR
                    )
                )
                print_ddp(blue('Added AugMix to transforms_list'))
                
        if 'RandAugment' in transform_dict:
            if transform_dict['RandAugment']['apply']:
                transform_list.append(
                    RandAugment(
                        num_ops=transform_dict['RandAugment']['num_ops'],
                        magnitude=transform_dict['RandAugment']['magnitude'],
                        # num_magnitude_bins=transform_dict['RandAugment']['num_magnitude_bins'],
                        interpolation=DEFAULT_INTERPOLATION_MODE
                    )
                )
                print_ddp(blue('Added RandAugment to transforms_list'))
        
        if 'AutoAugment' in transform_dict:
            if transform_dict['AutoAugment']['apply']:
                transform_list.append(
                    AutoAugment(
                        policy=AutoAugmentPolicy.IMAGENET,
                        interpolation=DEFAULT_INTERPOLATION_MODE,
                    )
                )
                print_ddp(blue('Added AutoAugment to transforms_list'))
        
        if 'TrivialAugment' in transform_dict:
            if transform_dict['TrivialAugment']['apply']:
                transform_list.append(
                    TrivialAugmentWide(
                        # num_magnitude_bins=transform_dict['TrivialAugment']['num_magnitude_bins'],
                        interpolation=DEFAULT_INTERPOLATION_MODE
                    )
                )
                print_ddp(blue('Added TrivialAugment to transforms_list'))
                
        
        # ----------------------------------------
        # third: final transformations: converting to tensor, normalize, and random erase
        # ----------------------------------------
        # Convert to Tensort
        transform_list.append(ToTensor())
        
        if "Normalize" in transform_dict:            
            if transform_dict['Normalize']:
                transform_list.append(Normalize(mean=self.mean, 
                                                std=self.std)) 
                
        if "RandomErasing" in transform_dict:                    
            if transform_dict['RandomErasing']['apply']:
                temp_d = transform_dict['RandomErasing']
                transform_list.append(RandomErasing(scale=temp_d['scale'],
                                                  ratio=temp_d['ratio'], 
                                                  value=temp_d['value'],
                                                  p=temp_d['p']))
                print_blue('Added RandomErasing to transforms_list')
        
        print_ddp(gray(f'{self.mode} TRANSFORMS LIST:'))
        print_ddp(gray(the_pprint.pformat(transform_list, sort_dicts=False) + '\n'))
        
        return transform_list
    
    def get_transform_defs(self):
        if self.mode == 'train':
            aplied_transforms = self.train_transforms
        if self.mode in ['val', 'eval']:
            aplied_transforms = self.val_transforms
        if self.mode == 'test':
            aplied_transforms = self.test_transforms
        return aplied_transforms   # this is a dict of transforms read from the param file
    
    def get_transforms(self):   
        applied_transforms = self.get_transform_defs()
        # SSL training
        if isinstance(applied_transforms, list):
            transforms = [Compose(self.get_trans_list(tr)) for tr in applied_transforms]
        # supervised training
        elif isinstance(applied_transforms, dict):
            transforms = Compose(self.get_trans_list(applied_transforms))
        else:
            raise TypeError("Transform data structure not understood")
        return self.__class__.disentangle_resizes_from_transforms(transforms)
    
    @staticmethod
    def remove_transform(old_transforms, transform_to_remove_type):            
        new_transforms = deepcopy(old_transforms)
        if isinstance(new_transforms, Compose):
            new_transforms = new_transforms.transforms
        
        new_transforms = [trans for trans in new_transforms 
                              if not isinstance(trans, transform_to_remove_type)]
        
        if isinstance(old_transforms, TwoCropTransfomCompose):
            return TwoCropTransfomCompose(new_transforms)
        return Compose(new_transforms)
    
    @staticmethod
    def disentangle_resizes_from_transforms(transforms):
        resizes = []
        resizing = None
        resize_free_trans = deepcopy(transforms)    
        
        # supervised training
        if isinstance(transforms, Compose):
            resizing = [ tr for tr in transforms.transforms if isinstance(tr, Resize)]
            resize_free_trans = BaseSet.remove_transform(resize_free_trans, Resize)
            resizing = None if not resizing else resizing[0]
            return resize_free_trans, resizing
        
        # SSL training
        elif isinstance(transforms, list):
            for ltr in transforms:
                resizes.append([tr for tr in ltr.transforms if isinstance(tr, Resize)][0])
            sizes = [tr.size for tr in resizes]
            # if all resizes are the same
            if len(set(sizes)) == 1 and len(sizes) > 1:
                resizing = deepcopy(resizes[0])
                resize_free_trans = [BaseSet.remove_transform(tr, Resize) for tr in resize_free_trans]
                return resize_free_trans, resizing
            # if we have different resizes return the original
            else:
                return transforms, resizing
        else:
            raise TypeError(f"Resize disentaglement does not support type {type(transforms)}")
    
    @staticmethod    
    def get_validation_ids(total_size, val_size, json_path, dataset_name, seed_n=42, overwrite=False):
        """ Gets the total size of the dataset, and the validation size (as int or float [0,1] 
        as well as a json path to save the validation ids and it 
        returns: the train / validation split)"""
        idxs = list(range(total_size))   
        if val_size < 1:
            val_size = int(total_size * val_size)  
        train_size = total_size - val_size

        if not os.path.isfile(json_path) or overwrite:
            print("Creating a new train/val split for \"{}\" !".format(dataset_name))
            random.Random(seed_n).shuffle(idxs)
            train_split = idxs[val_size:]
            val_split = idxs[:val_size]
            s_dict = {"train_split":train_split, "val_split":val_split}
            save_json(s_dict, json_path)    
        else:
            s_dict = load_json(json_path)
            if isinstance(s_dict, dict):
                val_split = s_dict["val_split"]
                train_split = s_dict["train_split"]
            elif isinstance(s_dict, list):
                val_split = s_dict
                train_split = list(set(range(total_size)) - set(val_split))
                assert len(train_split) + len(val_split) == total_size
            if val_size != len(val_split) or train_size != len(train_split):
                print("Found updated train/validation size for \"{}\" !".format(dataset_name))
                train_split, val_split = BaseSet.get_validation_ids(total_size, val_size, json_path, 
                                                          dataset_name, seed_n=42, overwrite=True)
        return train_split, val_split         
    
    
class BaseModel(nn.Module):
    """Base model that Classifier subclasses.
    
    This class only has utility functions like freeze/unfreeze and init_weights.
    Not intended to be used directly.
    """
    def __init__(self):
        super().__init__()  
        super().__init__() 
        self.use_mixed_precision = False        
        self.base_id = torch.cuda.current_device() if self.visible_world else "cpu"
    
    def attr_from_dict(self, param_dict):
        for key in param_dict:
            setattr(self, key, param_dict[key])
            
    def freeze_submodel(self, submodel=None):
        submodel = self if submodel is None else submodel
        for param in submodel.parameters():
            param.requires_grad = False
            
    def init_with_kaiming(self, submodel=None, init_type='normal'):
        submodel = self if submodel is None else submodel
        if init_type == 'uniform':
            weights_init = conv2d_kaiming_uniform_init
        elif init_type == 'normal':            
            weights_init = conv2d_kaiming_normal_init
        else:
            raise NotImplementedError
        submodel.apply(weights_init)             
                
    @property
    def visible_world(self):
        return torch.cuda.device_count()   
   
    @property
    def visible_ids(self):
        return list(range(torch.cuda.device_count()))
    
    @property
    def device_id(self):    
        did = torch.cuda.current_device() if self.visible_world else "cpu"
        assert self.base_id == did
        return did              
    
    @property
    def is_rank0(self):
        return is_rank0(self.device_id)
   
                
class BaseTrainer:
    """Base trainer class that Trainer subclasses.

    This class only has utility functions like save/load model.
    Not intended to be used directly.
    """
    def __init__(self):
        self.scaler = None
        self.use_mixed_precision = False        
        self.is_supervised = True
        self.val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.val_target = 0.
        self.best_val_target = 0.
        self.iters = 0
        self.epoch0 = 0
        self.epoch = 0
        self.base_id = torch.cuda.current_device() if self.visible_world else "cpu"
        # debugging
        self.is_debug = False  # does not log wandb
        self.is_dry = False  # logs wandb but does not save checkpoint
    
    def attr_from_dict(self, param_dict):
        for key in param_dict:
            setattr(self, key, param_dict[key])
            
    def load_session(self, restore_only_model=False, model_path=None):
        self.get_saved_model_path(model_path=model_path)
        if os.path.isfile(self.model_path) and self.restore_session:        
            print("Loading model from {}".format(self.model_path))
            checkpoint = torch.load(self.model_path)
            if is_parallel(self.model):
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            self.model.to(self.device_id)
            self.org_model_state = model_to_CPU_state(self.model)
            self.best_model = deepcopy(self.org_model_state)
            if self.scaler is not None and "scaler" in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler'])            
            if restore_only_model:
                return
            
            self.iters = checkpoint['iters']
            self.epoch = checkpoint['epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device_id)
            self.org_optimizer_state = opimizer_to_CPU_state(self.optimizer)
            print("Loaded checkpoint '{}' (epoch {})".format(self.model_path, checkpoint['epoch']))

        elif not os.path.isfile(self.model_path) and self.restore_session:
            print_byello(f"No checkpoint found at {self.model_path} during load_session."
                         "Assuming the checkpoint is directly loaded in wrapper using pretrained_path...")
    
    def get_saved_model_path(self, model_path=None):  # constructs model path based on save_dir and model_name
        if model_path is None:
            if not hasattr(self, "save_dir"):
                raise AttributeError("save_dir not found. Please specify the saving directory")
            # model_saver_dir = os.path.join(self.save_dir, 'checkpoints')
            model_saver_dir = self.save_dir
            check_dir(model_saver_dir)
            self.model_path = os.path.join(model_saver_dir, self.model_name) + '.pth'
        else:
            self.model_path = os.path.abspath(model_path) + '.pth'
        
    def save_session(self, model_path=None, verbose=True):
        if self.is_debug or self.is_dry:
            print_ddp('No model saving in debug/dry mode...')
        
        # we do not save model in dry/debug mode
        if self.is_rank0 and not (self.is_debug or self.is_dry):
            self.get_saved_model_path(model_path=model_path)
            if verbose:
                print("\nSaving model as {}\n".format(os.path.basename(self.model_path)) )
            state = {
                'iters': self.iters, 
                'state_dict': self.best_model, 
                'original_state' : self.org_model_state,
                'optimizer': opimizer_to_CPU_state(self.optimizer), 
                'epoch': self.epoch,
                'parameters' : self.parameters,
                'best_val_target': self.best_val_target
            }
            if self.scaler is not None:
                state['scaler'] = self.scaler.state_dict()            
            torch.save(state, self.model_path)
        synchronize()
        
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']
        
    def print_train_init(self):
        if self.is_rank0: 
            print("Start training with learning rate: {}".format(self.get_lr()))    
            
    def logging(self, logging_dict):
        if not self.is_rank0: return
        if wandb.run:  # if there is an active run
            wandb.log(logging_dict, step=self.iters)    
            
    def set_models_precision(self, use=False):
        if is_parallel(self.model):
            self.model.module.use_mixed_precision = use
        else:
            self.model.use_mixed_precision = use            
             
    @property
    def visible_world(self):
        return torch.cuda.device_count()   
   
    @property
    def visible_ids(self):
        return list(range(torch.cuda.device_count()))
    
    @property
    def device_id(self):    
        return torch.cuda.current_device() if self.visible_world else "cpu"
    
    @property
    def is_rank0(self):
        return is_rank0(self.device_id)
