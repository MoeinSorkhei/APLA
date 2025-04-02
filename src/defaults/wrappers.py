from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler as DS
from torch.utils.data import DataLoader
from torch import optim

from .models import *
from .datasets import *
from utils._utils import *
from utils.colors import *
from . import datasets


class DefaultWrapper:
    """Class that wraps everything.

    Model, optimizers, schedulers, and dataloaders are initialized in this class.

    Attributes:
        param_attributes:
            All the fields in the .json file are stored as attributes here.
    """
    def __init__(self, parameters: edict):
        """Inits the DefaultWrapper class.
        Args:
            parameters:
                Dictionary of paramaters read from a param file.
        """
        super().__init__()
        self.is_supervised = True  # default is supervised
        parameters = edict(deepcopy(parameters))
        parameters = self.update_augmentation_strategy(parameters)
        self.param_attributes = list(parameters.keys())
        
        for key in parameters:
            setattr(self, key, parameters[key])        
        
        self.train_collate_fn = None
        self.advanced_aug = hasattr(parameters.dataset_params.train_transforms, 'advanced_aug') \
                            and parameters.dataset_params.train_transforms.advanced_aug
        
    def instantiate(self):        
        """Initialize model, loss, metrics, dataloaders, optimizer and scheduler."""
        if self.is_rank0:
            print("Initialising Dataloaders . . .")
                    
        self.dataloaders = self.init_dataloaders()
        img_channels = self.dataloaders.trainloader.dataset.img_channels
        n_classes = self.dataloaders.trainloader.dataset.n_classes
        knn_nhood = self.dataloaders.trainloader.dataset.knn_nhood
        target_metric = self.dataloaders.trainloader.dataset.target_metric
        print_ddp(f"The default metric has been set to : \033[94m{target_metric}\033[0m")
        
        self.model_params.img_channels = img_channels
        self.model_params.knn_nhood = knn_nhood
        self.model_params.target_metric = target_metric
        
        # Checking for binary multi-label
        self.model_params.n_classes = n_classes
        is_multiclass = self.dataloaders.trainloader.dataset.is_multiclass
        if not is_multiclass and n_classes <= 2:
            print("\033[93m Binary multi-label problem found: CHANING THE n_classes to 1\033[0m")
            self.model_params.n_classes = 1
        
        # init and get model
        print_ddp("Initialising Model . . .")     
        self.model = self.init_model()  
        
        # init and get optimizer
        print_ddp("Initialising Optimization methods . . ")                
        optimizer_defs = self.init_optimizer(self.model, self.optimization_params.default)  
        self.attr_from_dict(optimizer_defs)
        
        # init and get scheduler
        epochs = self.training_params.epochs
        scheduler_defs = self.init_scheduler(self.optimizer,
                                              self.optimization_params.default, 
                                              len(self.dataloaders.trainloader), 
                                              epochs)  
        self.schedulers = MixedLRScheduler(**scheduler_defs)
        
        # init loss functions
        self.criterion = self.init_criteria()  
        
        # init metric functions
        self.init_metrics()
        
    def init_dataloaders(self) -> edict:
        """Define dataset params and dataloaders.
        
        Args:
            collate_fn:
                Specific collate_fn for the torch.utils.data.DataLoader.
        
        Returns:
            A dict (EasyDict) with train, validation and test loaders. nonddp_trainloader is
            for the 2nd phase of SWAP training where we don't use the distributed sampler.
            
            {'trainloader': trainloader,
             'valloader': valloader,
             'testloader': testloader,
             'nonddp_trainloader':nonddp_trainloader}
        """ 
        feature_bank_set, feature_bank_Loader = None, None
        DataSet = getattr(datasets, self.dataset_params.dataset)  # return dataset class

        trainset = DataSet(self.dataset_params, mode='train')
        valset   = DataSet(self.dataset_params, mode='val')
        testset  = DataSet(self.dataset_params, mode='test')

        if self.training_params.knn_eval or not self.is_supervised:
            feature_bank_set = DataSet(self.dataset_params, mode='train')
            feature_bank_set.transform = valset.transform   # Use validation transform when setting up prototype vectors
            feature_bank_set.resizing = valset.resizing 
            feature_bank_set.is_multi_crop = False
        
        if not self.is_supervised:
            trainset.num_augmentations = 2 
        
        # register task defs
        self.task = trainset.task
        self.is_multiclass = trainset.is_multiclass        
        
        train_sampler = None
        feature_bank_sampler = None
        train_shuffle = self.dataloader_params['trainloader']['shuffle']
        
        # creating distributed samplers 
        if ddp_is_on():        
            train_sampler = DS(trainset, num_replicas=self.visible_world, rank=self.device_id)  # shuffle is True
            if feature_bank_set is not None:
                feature_bank_sampler = DS(feature_bank_set, num_replicas=self.visible_world, shuffle=False, rank=self.device_id)
            # making shuffle explicitly False to avoid conflict with Distributed Sampler
            # This way the shuffling in trainLoader comes from train_sampler only
            self.dataloader_params['trainloader']['shuffle'] = False

        # define distributed samplers etc
        if self.advanced_aug:  # mixup etc.
            adv_aug_params = self.parameters.dataset_params.train_transforms.advanced_aug_params
            adv_aug_params.num_classes = trainset.n_classes
            self.train_collate_fn = AdvancedAugCollate(adv_aug_params)
        
        trainLoader = DataLoader(trainset, **self.dataloader_params['trainloader'], sampler=train_sampler, collate_fn=self.train_collate_fn)
        testLoader  = DataLoader(testset, **self.dataloader_params['testloader'])
        if len(valset) > 0 :
            valLoader   = DataLoader(valset, **self.dataloader_params['valloader'])
        else:
            valLoader = testLoader
        
        if feature_bank_set is not None:
            data_params_copy_feature_bank = deepcopy(self.dataloader_params['valloader'])
            data_params_copy_feature_bank['shuffle'] = False
            feature_bank_Loader = DataLoader(feature_bank_set, **data_params_copy_feature_bank, sampler=feature_bank_sampler)
        
        # restoring its original value
        self.dataloader_params['trainloader']['shuffle'] = train_shuffle

        if not len(valLoader):
            valLoader = testLoader            
            if self.is_rank0:
                warnings.warn("Warning... Using test set as validation set")

        return edict({
            'trainloader': trainLoader,
            'valloader': valLoader,
            'testloader': testLoader,
            'fbank_loader': feature_bank_Loader,
        })
        

    def init_model(self) -> Classifier:
        """Initialize the model.
        DDP broadcasts model states from rank 0 process to all other processes 
        in the DDP constructor, you do not need to worry about different DDP processes 
        start from different model parameter initial values.   
        """
        model = Classifier(model_params=self.model_params, system_params=self.system_params)
        if self.transfer_learning_params.pretrained_path:   # load the checkpoint using pretrained_path
            print_blue(f'Wrapper init_model. Loading pretrained model from path: {self.transfer_learning_params.pretrained_path}')
            load_from_pretrained(model, self.transfer_learning_params.pretrained_path, strict=True)    
            
        model.to(self.device_id)
        if self.visible_world > 1 and torch.distributed.is_initialized():
            model = DDP(model, device_ids=[self.device_id])
        return model
    
    @staticmethod
    def init_optimizer(model, optimization_params:edict) -> edict:    
        """Initialize the optimizer.
        
        Args:
            optimization_params: EasyDict instance, read from the .json file.

        Returns:
            A dict (EasyDict) with optimizer and type keys.
            {'optimizer': optimizer (e.g. a torch.optim.Adam instance),
             'optimizer_type': optimizer_type (e.g. a string "Adam")}
        """
        optimizer_type = optimization_params.optimizer.type
        opt = optim.__dict__[optimizer_type]
        opt_params = optimization_params.optimizer.params
        optimizer = opt(DefaultWrapper.get_params_groups(model), **opt_params)
        return edict({"optimizer":optimizer, "optimizer_type":optimizer_type})
    
    @staticmethod
    def get_params_groups(model):
        """
        FROM: https://github.com/facebookresearch/dino/blob/main/utils.py
        It filters-out the no-grad params and it excludes weight_decay from all non-weight / non-bias tensors
        It will return 2 groups 0: regularized 1: not_regularized
        """
        regularized = []
        not_regularized = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)
        return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]     
        
    @staticmethod        
    def init_scheduler(optimizer, optimization_params: edict, steps_per_epoch: int=None, epochs: int=None) -> edict:          
        """Initialize the learning rate scheduler.

        steps_per_epoch and epochs are set by the caller, they are not intended to be None.
        
        Args:
            optimization_params: EasyDict instance, read from the .json file.
        
        Returns:
            A dict (EasyDict) with scheduler and type keys.
            {'scheduler': scheduler (e.g. a torch.optim.lr_scheduler.OneCycleLR instance),
             'scheduler_type': scheduler_type (e.g. a string "OneCycleLR")}
        """
        schedulers = edict({"schedulers":[None], "scheduler_types":[None], 
                           "steps_per_epoch":steps_per_epoch})
        scheduler_types = optimization_params.scheduler.type
        accepted_types = [
            None, 
            "LinearWarmup", 
            "MultiStepLR", 
            "ReduceLROnPlateau", 
            "OneCycleLR",
            "CosineAnnealingLR",
            "PolynomialLR"
        ] 
        if not isinstance(scheduler_types, list):
            scheduler_types = [scheduler_types]        
        
        for scheduler_type in scheduler_types:
            if scheduler_type not in accepted_types:
                raise ValueError(f"{scheduler_type} is not a supported scheduler")
            
            if scheduler_type is None:
                continue
            elif scheduler_type not in optim.lr_scheduler.__dict__:
                if scheduler_type == 'LinearWarmup':
                    sch = LinearWarmup
                elif scheduler_type == 'PolynomialLR':
                    sch = PolynomialLR
                else:
                    raise NotImplementedError
            else:
                sch = optim.lr_scheduler.__dict__[scheduler_type]

            if sch.__name__ == 'OneCycleLR':
                max_lr = optimization_params.optimizer.params.lr
                sch_params = {"max_lr":max_lr, 
                              "steps_per_epoch":steps_per_epoch, 
                              "epochs":epochs,
                              "div_factor": max_lr/1e-8
                             }
                if "LinearWarmup" in scheduler_types:
                    sch_params["div_factor"] = 1.
                sch_params.update(optimization_params.scheduler.params.OneCycleLR)
            
            elif sch.__name__ == 'LinearWarmup':
                max_lr = optimization_params.optimizer.params.lr
                sch_params = optimization_params.scheduler.params[scheduler_type]
                sch_params.update({"max_lr":max_lr, "steps_per_epoch":steps_per_epoch})
            
            elif sch.__name__ == 'CosineAnnealingLR':
                T_max = steps_per_epoch * epochs
                sch_params = optimization_params.scheduler.params[scheduler_type]
                if "LinearWarmup" in scheduler_types:
                    T_max = T_max - warmup_iters
                sch_params.update({"T_max":T_max})
            
            elif sch.__name__ == 'PolynomialLR':
                T_max = steps_per_epoch * epochs
                sch_params = optimization_params.scheduler.params[scheduler_type]
                if "LinearWarmup" in scheduler_types:
                    T_max = T_max - warmup_iters
                sch_params.update({"total_iters": T_max})
            
            else:
                sch_params = optimization_params.scheduler.params[scheduler_type]
            
            scheduler = sch(optimizer, **sch_params) 
            schedulers["schedulers"].append(scheduler)
            schedulers["scheduler_types"].append(scheduler_type)
            
            if scheduler_type == 'LinearWarmup':
                warmup_iters = scheduler.warmup_iters

        return schedulers
    
    def init_criteria(self):          
        """Initialize the loss criteria.  """
        if self.task == 'classification':
            if self.is_multiclass:
                crit = nn.CrossEntropyLoss() 
            else:
                crit = nn.BCEWithLogitsLoss() 
        else:
            raise NotImplementedError("Only classification tasks are implemented for now")
            
        return crit
    
    def init_metrics(self):
        if self.task == 'classification':
            if self.is_multiclass:
                self.metric = ClassificationMetrics
            else:
                self.metric = MultiLabelClassificationMetrics
        else:
            raise NotImplementedError("Only classification tasks are implemented for now")    
    
    def attr_from_dict(self, param_dict: edict):
        """Function that makes the dictionary key-values into attributes.
        
        This allows us to use the dot syntax. Check the .json file for the entries.

        Args:
            param_dict: The dict we populate the class attributes from.
        """
        self.name = self.__class__.__name__
        for key in param_dict:
            setattr(self, key, param_dict[key])   
            
    def update_augmentation_strategy(self, parameters):
        """
        Overrides augmentation strategies, used for self-supervision.
        """
        self_dir = os.path.dirname(os.path.abspath(inspect.getfile(self.__class__)))  # the current dir where wrapper.py is
        new_strategy_filepath = os.path.join(self_dir, "augmentation_strategy.json")
        
        # only for SSL: dino etc.
        if not os.path.isfile(new_strategy_filepath):
            return parameters
        
        print_ddp(green(f'Using aug strategy file: {new_strategy_filepath}'))
        
        augmentation_strategy = edict(load_json(new_strategy_filepath))
        general_args = augmentation_strategy.general_args
        repetition_strategy = augmentation_strategy.repetition_strategy
        transforms = augmentation_strategy.transforms
        to_change = list(transforms.keys())
        
        if not general_args.overwrite_defaults:
            return parameters
        params = deepcopy(parameters)
        
        for org_keys in parameters.dataset_params.keys():
            if org_keys in to_change:
                org_def = parameters.dataset_params[org_keys]
                updated_transforms = []
                for order, aug_type in enumerate(repetition_strategy.order):
                    new_trans = transforms[org_keys][aug_type]
                    n_augs = repetition_strategy.n_augmentations[order]
                    if general_args.inherit:
                        for key in general_args.inherit:
                            new_trans[key] = org_def[key]                    
                    for _ in range(n_augs):
                        updated_transforms.append(new_trans)
                params.dataset_params[org_keys] = updated_transforms                    
        return params            
        
    @property
    def parameters(self):
        return edict({key : getattr(self, key) 
                      for key in self.param_attributes})
    
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
