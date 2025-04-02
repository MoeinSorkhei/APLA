from self_supervised.BYOL.trainer import *
from self_supervised.DINO.trainer import *
from .dinov2_utils import CosineScheduler
from utils.colors import *


def build_schedulers(optim_params, training_params, teacher_params, iters_per_epoch, total_iters):
    assert optim_params.scheduler.type == ["LinearWarmup", "CosineAnnealingLR"]  # only this implemented for now
    # start_warmup_value -> (LINEAR) -> base_value -> (COSINE) -> final_value
    lr = dict(  # warm-up + cosine
        start_warmup_value=0,
        base_value=optim_params.optimizer.params.lr,
        final_value=optim_params.scheduler.params.CosineAnnealingLR.eta_min,
        total_iters=total_iters,
        warmup_iters=optim_params.scheduler.params.LinearWarmup.warmup_epochs * iters_per_epoch
    )
    wd = dict(
        base_value=optim_params.optimizer.params.weight_decay,
        # final_value=optim_params.params.weight_decay_end,
        final_value=1e-4,  # hard-coded
        total_iters=total_iters,
        warmup_iters=0  # no warm-up, just cosine
    )
    momentum = dict(
        base_value=teacher_params.momentum_teacher,
        final_value=teacher_params.final_momentum_teacher,
        total_iters=total_iters,
        warmup_iters=0  # no warm-up, just cosine
    )
    teacher_temp = dict(  # only linear warmup, then remains constant (no cosine)
        start_warmup_value=teacher_params.warmup_teacher_temp,
        base_value=teacher_params.teacher_temp,
        final_value=teacher_params.teacher_temp,
        total_iters=teacher_params.warmup_teacher_temp_epochs * iters_per_epoch,
        warmup_iters=teacher_params.warmup_teacher_temp_epochs * iters_per_epoch
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[
        : training_params.freeze_last_layer_epochs * iters_per_epoch
    ] = 0  # mimicking the original schedules

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


class Dinov2Trainer(DINOTrainer):
    def __init__(self, wrapper):
        super().__init__(wrapper) 
        self.wrapper = wrapper
        self.mixed_eval = False
        self.best_test_target = -np.inf  # for TTT
        # self.eval_test = True
        self.freeze_last_for = wrapper.training_params.freeze_last_layer_epochs
        
        self.iters_per_epoch = len(self.wrapper.dataloaders.trainloader)  # n_imgs / (num_gpu * bsize_per_gpu)
        self.total_iters = self.iters_per_epoch * self.wrapper.training_params.epochs
        (
            self.lr_schedule,
            self.wd_schedule,
            self.momentum_schedule,
            self.teacher_temp_schedule,
            self.last_layer_lr_schedule
        ) = build_schedulers(optim_params=self.wrapper.optimization_params.default, 
                             training_params=self.wrapper.training_params,
                             teacher_params=self.wrapper.model_params.transformers_params.teacher,
                             iters_per_epoch=self.iters_per_epoch, 
                             total_iters=self.total_iters)
        
    @property
    def is_ttt(self):
        return hasattr(self.wrapper.dataset_params, 'is_ttt') and self.wrapper.dataset_params.is_ttt
    
    def possibly_cancel_last_layer_grads(self, verbose=False):
        if self.freeze_last_for and self.epoch <= self.freeze_last_for:
            cancel_gradients(self.model, "student.dino_head.last_layer")
            cancel_gradients(self.model, "student.ibot_head.last_layer")
            if verbose:
                print(byello(f'Canceled grads of last layer in epoch: {self.epoch}'))
    
    def apply_optim_scheduler(self, lr, wd, verbose=False):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.optimizer.param_groups[0]["weight_decay"] = wd  # group 1 is not non-regularized params
        # -- print
        if verbose:
            for i, param_group in enumerate(self.optimizer.param_groups):
                print_ddp(byello(f'param gorup {i}: lr: {param_group["lr"]} -- wd: {param_group["weight_decay"]}'))
    
    def get_lr(self, param_group_indx=0):
        return self.optimizer.param_groups[param_group_indx]['lr']
    
    def get_wd(self, param_group_indx):
        return self.optimizer.param_groups[param_group_indx]['weight_decay']
    
    def global_step(self, **kwargs):
        data = kwargs['batch']  # kwargs coming from BYOL train function
        
        # --- apply schedules
        lr = self.lr_schedule[self.iters]
        wd = self.wd_schedule[self.iters]
        teacher_temp = self.teacher_temp_schedule[self.iters]
        mom = self.momentum_schedule[self.iters]
        self.apply_optim_scheduler(lr=lr, wd=wd)
        
        # --- go through the model
        self.optimizer.zero_grad()
        with autocast(self.use_mixed_precision):
            loss, loss_dict = self.model(images=data['images'], teacher_temp=teacher_temp)
                
        # --- backprop
        if not self.use_mixed_precision:
            loss.backward() 
            if self.grad_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clipping)
            self.possibly_cancel_last_layer_grads()
            self.optimizer.step()  
        else:
            self.scaler.scale(loss).backward()
            if self.grad_clipping:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clipping)
            self.possibly_cancel_last_layer_grads()
            self.scaler.step(self.optimizer)
            self.scaler.update() 
        
        # --- update teacher with momentum
        if ddp_is_on():
            # assuming that all models are at the same state 
            # Otherwise this is wrong and we need to synchronise the weights first!!!!
            self.model.module.update_teacher(mom)
        else:
            self.model.update_teacher(mom)

        # --- logging
        if self.iters % self.log_every == 0 or (self.iters == 1):
            loss = dist_average_tensor(loss)
            if self.is_rank0:
                self.logging({
                    **{
                        'train_' + key: value for key, value in loss_dict.items()
                    }, 
                    **{
                        'train_loss': loss.item(),
                        'teacher_temp': self.teacher_temp_schedule[self.iters],
                        'momentum': self.momentum_schedule[self.iters],
                        'lr_group_0': self.get_lr(param_group_indx=0),
                        'lr_group_1': self.get_lr(param_group_indx=1),
                        'wd_0': self.get_wd(param_group_indx=0),
                        'wd_1': self.get_wd(param_group_indx=1)
                    }
                })

    @property
    def feature_extractor(self):
        return DINOv2_to_classifier(self.model)

                    
def DINOv2_to_classifier(net):
    if is_parallel(net):
        return net.module.teacher.backbone
    else:
        return net.teacher.backbone
