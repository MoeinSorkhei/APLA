import pprint as the_pprint
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data.distributed import DistributedSampler as DS
from datetime import datetime
import torch.nn.functional as F

import utils
from defaults.bases import *
from utils.colors import *


class Trainer(BaseTrainer):
    """Main trainer class.

    Initializes with a DefaultWrapper instance as its input. 
    Call trainer.train() to train and validate or call trainer.test()
    Training with standard DDP: a model is trainedon multiple machines/gpus using distributed gradients. 
    """
    def __init__(self, wraped_defs):
        """Initialize the trainer instance.
        
        This function clones its attributes from the DefaultWrapper instance or generates
        them from the .json file. 
        """
        super().__init__()
        self.wrapped_defs = wraped_defs
        self.parameters = wraped_defs.parameters
        self.is_supervised = wraped_defs.is_supervised        
        self.training_params = self.parameters.training_params
        self.attr_from_dict(self.training_params)
        self.attr_from_dict(wraped_defs.dataloaders)
        self.epoch_steps = len(self.trainloader)
        self.total_steps = int(len(self.trainloader) * self.epochs)
        
        self.model = wraped_defs.model
        self.criterion = wraped_defs.criterion        
        self.optimizer = wraped_defs.optimizer 
        self.scheduler = wraped_defs.schedulers
        self.metric_fn = wraped_defs.metric
        self.advanced_aug = wraped_defs.advanced_aug
        
        # print some info
        print_ddp('\n\n')
        print_ddp(f'Trainer -- Using advance_aug: {self.advanced_aug}')
        print_ddp(f'Trainer -- wrapper train_collate_fn: {self.wrapped_defs.train_collate_fn}')
        print_ddp(f'Full wrapper definition:')
        # print_ddp(gray(the_pprint.pformat(self.wrapped_defs.__dict__, sort_dicts=False)))
        print_ddp('\n\n')
        
        # model states etc
        self.org_model_state = model_to_CPU_state(self.model)
        self.org_optimizer_state = opimizer_to_CPU_state(self.optimizer)
        self.total_step = len(self.trainloader)        
        self.best_model = deepcopy(self.org_model_state)
        
        # mixed precision 
        if self.use_mixed_precision:
            self.scaler = GradScaler()
            self.set_models_precision(self.use_mixed_precision)        
    
    def train(self, verbose=False):
        """Main training loop."""
        self.test_mode = False
        self.load_session(self.restore_only_model)
        self.print_train_init()
        
        n_classes = self.trainloader.dataset.n_classes
        metric = self.metric_fn(n_classes=n_classes, mode="train")
        epoch_bar = range(self.epoch0 + 1, self.epoch0 + self.epochs + 1)
        if self.is_rank0:
            epoch_bar = tqdm(epoch_bar, desc='Epoch', leave=False)
            
        for self.epoch in epoch_bar:            
            self.model.train() 
            if isinstance(self.trainloader.sampler, DS):
                self.trainloader.sampler.set_epoch(self.epoch)            
            
            iter_bar = enumerate(self.trainloader)
            if self.is_rank0:
                iter_bar = tqdm(iter_bar, desc='Training', leave=False, total=len(self.trainloader))
            for it, batch in iter_bar:
                self.iters += 1
                self.global_step(batch=batch, metric=metric, it=it)   
                
                if self.val_every != np.inf:
                    if self.iters % int(self.val_every * self.epoch_steps) == 0: 
                        synchronize()
                        self.epoch_step()  
                        self.model.train()
                        if verbose:
                            print(utils.get_nvidia_smi_output(gpu=self.wrapped_defs.system_params.which_GPUs))
                synchronize()
                if verbose:
                    print(f"\nIter: {self.iters:,} done at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
            if not self.save_best_model:
                self.best_model = model_to_CPU_state(self.model)   
                self.save_session(verbose=True)                         
                
        if self.is_rank0:         
            print(" ==> Training done")
        self.evaluate()
        self.save_session(verbose=True)
        synchronize()
        
    def global_step(self, **kwargs):
        """Function for the standard forward/backward/update.
        
        If using DDP, metrics (e.g. accuracy) are calculated with dist.all_gather
        """
        self.optimizer.zero_grad()
        
        metric = kwargs['metric']        
        images, labels = kwargs['batch']
        if len(labels) == 2 and isinstance(labels, list):
            ids    = labels[1]
            labels = labels[0]
        labels = labels.to(self.device_id, non_blocking=True)
        images = images.to(self.device_id, non_blocking=True) 
        
        # forward
        with autocast(self.use_mixed_precision):
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

        # backward
        if not self.use_mixed_precision:
            loss.backward() 
            if self.grad_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clipping)
            self.optimizer.step()  
        else:
            self.scaler.scale(loss).backward()
            if self.grad_clipping:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clipping)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        if not self.advanced_aug:
            metric.add_preds(outputs, labels)  # distributed gather inside
        
        self.scheduler.step(self.val_target, self.val_loss)

        if self.iters % self.log_every == 0 or self.iters == 1:
            loss = dist_average_tensor(loss)
            if self.is_rank0:
                self.logging({'train_loss': loss.item(), 'learning_rate': self.get_lr()})
                if not self.advanced_aug:
                    self.logging(metric.get_values())     
                    metric.reset()                
    
    def epoch_step(self): 
        """Function for periodic validation, LR updates and model saving.
        
        Note that in the 2nd phase of training, the behavior is different, each model on
        each GPU is saved separately.
        """
        self.evaluate()        
        self.save_session(verbose=True)                  
    
    def evaluate(self, dataloader=None):
        """Validation loop function.
        
        This is pretty much the same thing with global_step() but with torch.no_grad()
        Also note that DDP is not used here. There is not much point to DDP, since 
        we are not doing backprop anyway.
        """
        
        # if we are using knn evaluations build a feature bank of the train set
        if self.knn_eval:
            self.build_feature_bank()
            
        if not self.is_rank0: return
        self.model.eval()

        if dataloader == None:
            dataloader = self.valloader

        if not len(dataloader):
            self.best_model = model_to_CPU_state(self.model)
            self.model.train()
            return
        
        knn_nhood = dataloader.dataset.knn_nhood
        n_classes = dataloader.dataset.n_classes
        target_metric = dataloader.dataset.target_metric
        
        if self.is_rank0:
            metric = self.metric_fn(n_classes=n_classes, mode="val")
            if self.knn_eval:
                knn_metric = self.metric_fn(n_classes=n_classes, mode="knn_val", raw=False)
            iter_bar = tqdm(dataloader, desc='Validating', leave=False, total=len(dataloader))
        else:
            iter_bar = dataloader

        val_loss = []
        feature_bank = []
        with torch.no_grad():
            for images, labels in iter_bar:
                if len(labels) == 2 and isinstance(labels, list):
                    ids    = labels[1]
                    labels = labels[0]
                labels = labels.to(self.device_id, non_blocking=True)
                images = images.to(self.device_id, non_blocking=True)

                if is_ddp(self.model):
                    outputs, features = self.model.module(images, return_embedding=True)
                else:
                    outputs, features = self.model(images, return_embedding=True)
                    
                if self.knn_eval:
                    features = F.normalize(features, dim=1)
                    pred_labels = self.knn_predict(feature = features, 
                                                   feature_bank=self.feature_bank, 
                                                   feature_labels= self.targets_bank, 
                                                   knn_k=knn_nhood, knn_t=0.1, classes=n_classes, 
                                                   multi_label = not dataloader.dataset.is_multiclass)
                    knn_metric.add_preds(pred_labels, labels)

                loss = self.criterion(outputs, labels)
                val_loss.append(loss.item())
                metric.add_preds(outputs, labels)
                
        self.val_loss = np.array(val_loss).mean()
        eval_metrics = metric.get_values(use_dist=isinstance(dataloader,DS))
        if self.knn_eval:
            eval_metrics.update(knn_metric.get_values(use_dist=isinstance(dataloader,DS)))
        self.val_target = eval_metrics[f"val_{target_metric}"]

        self.logging(eval_metrics)
        self.logging({'val_loss': round(self.val_loss, 5)})
        
        if self.val_target > self.best_val_target:
            self.best_val_target = self.val_target
            if self.save_best_model:
                self.best_model = model_to_CPU_state(self.model)
        
        if self.val_loss <= self.best_val_loss:
            self.best_val_loss = self.val_loss
        
        if not self.save_best_model:
            self.best_model = model_to_CPU_state(self.model)
        
        self.model.train()
        
    def test(self, dataloader=None, chpt_path=None):
        """Test function.
        
        Just be careful you are not explicitly passing the wrong dataset here.
        Otherwise it will use the test set.
        """
        if not self.is_rank0: return
            
        self.test_mode = True
        self.restore_session = True
        self.restore_only_model = True
        self.set_models_precision(False)
        
        # two ways of loading checkpoint:
        # 1 when chpt_path is provided (e.g. when inference with model, or using a specific checkpoint)
        # 2 in load_session, which uses save_dir and model_name (e.g. testing at the end of training)
        if chpt_path:
            print_blue(f'Doing test, using the provided chpt_path: {chpt_path}')
            load_from_pretrained(self.model, chpt_path, strict=True)
        else:
            print_blue(f'Doing test, using the load_session, since chpt_path not provided')
            self.load_session(self.restore_only_model)
        
        self.model.eval()
        
        if self.knn_eval:
            self.build_feature_bank()
            print(f'Doing test, building feature bank done')
            
        if dataloader == None:
            dataloader = self.testloader  
        
        # test metrics
        test_loss = []
        knn_nhood = dataloader.dataset.knn_nhood
        n_classes = dataloader.dataset.n_classes    
        if self.is_supervised:
            metric = self.metric_fn(n_classes=n_classes, mode="test")
        if self.knn_eval or not self.is_supervised:
            knn_metric = self.metric_fn(n_classes=n_classes, mode="knn_test", raw=False)
        iter_bar = tqdm(dataloader, desc='Testing', leave=True, total=len(dataloader))
        
        with torch.no_grad():
            for images, labels in iter_bar: 
                if len(labels) == 2 and isinstance(labels, list):
                    _      = labels[1]
                    labels = labels[0]
                labels = labels.to(self.device_id, non_blocking=True)
                images = images.to(self.device_id, non_blocking=True)                   
                
                if is_ddp(self.model):
                    outputs, features = self.model.module(images, return_embedding=True)
                else:
                    outputs, features = self.model(images, return_embedding=True)
                    
                if self.knn_eval:
                    features = F.normalize(features, dim=1)
                    pred_labels = self.knn_predict(feature = features, 
                                                   feature_bank=self.feature_bank, 
                                                   feature_labels= self.targets_bank, 
                                                   knn_k=knn_nhood, knn_t=0.1, classes=n_classes,
                                                   multi_label = not dataloader.dataset.is_multiclass)
                    knn_metric.add_preds(pred_labels, labels) 
                
                if self.is_supervised:
                    loss = self.criterion(outputs, labels)
                    test_loss.append(loss.item())
                    metric.add_preds(outputs, labels)
                
        self.test_loss = np.array(test_loss).mean() if test_loss else None
        test_metrics = {}
        
        if self.is_supervised:
            test_metrics = metric.get_values(use_dist=isinstance(dataloader, DS))
        
        if self.knn_eval or not self.is_supervised:
            knn_test_metrics = {}
            knn_test_metrics = knn_metric.get_values(use_dist=isinstance(dataloader, DS))
        
        if self.is_supervised:
            test_metrics['test_loss'] = round(self.test_loss, 5)
        
        self.model.train()
        self.set_models_precision(self.use_mixed_precision)
        
        def print_metrics_table(metrics):
            metrics_table = pd.DataFrame.from_dict(metrics, orient='index').T
            print(tabulate(metrics_table, headers='keys', tablefmt='psql'))
    
        print('\n', "--" * 5, f"Evaluation done", "--" * 5, '\n')
        print_metrics_table(test_metrics)
        if self.knn_eval or not self.is_supervised:
            print_metrics_table(knn_test_metrics)
        
        if wandb.run:  # if there is an active run
            self.logging(test_metrics)
            print(f'Logged test results at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        
        return test_metrics

    def build_feature_bank(self, dataloader=None, **kwargs):
        """Build feature bank function.
        
        This function is meant to store the feature representation of the training images along with their respective labels 

        This is pretty much the same thing with global_step() but with torch.no_grad()
        Also note that DDP is not used here. There is not much point to DDP, since 
        we are not doing backprop anyway.
        """
        
        self.model.eval()
        if dataloader is None:
            dataloader = self.fbank_loader         
        
        if self.is_rank0:
            iter_bar = tqdm(dataloader, desc='Building Feature Bank', leave=False, total=len(dataloader))
        else:
            iter_bar = dataloader
            
        with torch.no_grad():
            
            self.feature_bank = []
            self.targets_bank = []   
            for images, labels in iter_bar:
                if len(labels) == 2 and isinstance(labels, list):
                    ids    = labels[1]
                    labels = labels[0]
                labels = labels.to(self.device_id, non_blocking=True)
                images = images.to(self.device_id, non_blocking=True)                   
                
                if is_ddp(self.model):
                    _, feature = self.model.module(images, return_embedding=True)
                else:
                    _, feature = self.model(images, return_embedding=True)
                  
                feature = F.normalize(feature, dim=1)
                self.feature_bank.append(feature)
                self.targets_bank.append(labels)

            self.feature_bank = torch.cat(self.feature_bank, dim=0).t().contiguous()
            self.targets_bank = torch.cat(self.targets_bank, dim=0).t().contiguous()

            synchronize()
            self.feature_bank = dist_gather(self.feature_bank, cat_dim=-1)
            self.targets_bank = dist_gather(self.targets_bank, cat_dim=-1)
        self.model.train()
        
    def knn_predict(self, feature, feature_bank, feature_labels, 
                    knn_k: int, knn_t: float, classes: int = 10, multi_label = False):
        """Helper method to run kNN predictions on features based on a feature bank

        Args:
            feature: Tensor of shape [N, D] consisting of N D-dimensional features
        feature_bank: Tensor of a database of features used for kNN
        feature_labels: Labels for the features in our feature_bank
        classes: Number of classes (e.g. 10 for CIFAR-10)
        knn_k: Number of k neighbors used for kNN
        knn_t:

        """
        if multi_label:
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            # Find the similarities between the batch samples and the feature bank
            sim_matrix = torch.mm(feature, feature_bank)
            # identify the knn_k most similar samples in the feature bank for each of the batch samples
            sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
            # Expand the feature labels to a have a copy per batch sample
            expanded_labels = feature_labels.expand((feature.size(0),feature_labels.size(0),feature_labels.size(1)))

            # Unsqueeze and expand the similarity indicies and weights 
            sim_indices = sim_indices.unsqueeze_(1)
            sim_weight  = sim_weight.unsqueeze_(1)
            sim_indices_expanded = sim_indices.expand((sim_indices.size(0),feature_labels.size(0),sim_indices.size(2)))
            sim_weight_expanded  =  sim_weight.expand((sim_weight.size(0) ,feature_labels.size(0), sim_weight.size(2)))

            # Gather the labels of the most similar samples in the feature bank
            gathered = torch.gather(expanded_labels, dim=-1, index=sim_indices_expanded)

            # Scale the weights of the most similar samples
            sim_weight_expanded = (sim_weight_expanded / knn_t).exp()

            # weight each of the labels 
            weighted_labels = F.normalize(sim_weight_expanded,p=1,dim=2)*gathered
            pred_labels = weighted_labels.sum(axis=2)
            
            return pred_labels
        
        else:
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)

            # we do a reweighting of the similarities
            sim_weight = (sim_weight / knn_t).exp()

            # counts for each class
            one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
            # convert them to probablilities
            pred_scores = pred_scores/pred_scores.sum(1).unsqueeze(1)
            #pred_labels = pred_scores.argsort(dim=-1, descending=True)[:, 0]
            
        return pred_scores
