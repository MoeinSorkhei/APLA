#!/usr/bin/env python
# coding: utf-8

import os
import wandb
import argparse
import torch
import pprint
import glob
from datetime import datetime

from defaults import *
from self_supervised import *
from utils.system_def import *
from utils.launch import dist, launch, synchronize


def parse_arguments():
    parser = argparse.ArgumentParser(description='The main takes as argument the parameters dictionary from a json file', allow_abbrev=False)
    parser.add_argument('--params_path', type=str, required=False, help='Give the path of the json file which contains the training parameters')
    
    # train args
    parser.add_argument('--gpu', type=str, required=False, help='The GPU to be used for this run')
    parser.add_argument('--batch_size', type=int, help='Change the batch size of all data loaders')
    parser.add_argument('--val_every', type=float, help='How many epochs between each validation')
    parser.add_argument('--log_every', type=int)
    parser.add_argument('--mixed_precision', action='store_true', default=False)
    parser.add_argument('--num_workers', type=str)  # tr so that "0" takes effect as well
    parser.add_argument('--prefetch_factor', type=str)  # str so that "0" takes effect as well
    parser.add_argument('--lr', type=float)
    parser.add_argument('--warmup', type=int)
    parser.add_argument('--epochs', type=int, required=False)
    parser.add_argument('--wd', type=float)
    parser.add_argument('--dpr', type=float)  # drop path rate
    parser.add_argument('--dr', type=float)  # drop rate (mlp, attn proj dropout)
    parser.add_argument('--adr', type=float)  # attn drop rate (qkv dropout)
    
    # general run args
    parser.add_argument('--model_name', type=str, required=False, help='Used to manipulate the model_name defined in the param file for this run')
    parser.add_argument('--pretrained_path', type=str)
    parser.add_argument('--save_dir', type=str, required=False, help='Change the "save_dir" in param file')
    parser.add_argument('--debug', action='store_true', default=False, help='Flag for turning on the debug_mode')
    parser.add_argument('--dry', action='store_true', default=False, help='Flag for changing parm file suitable for a dry run')
    parser.add_argument('--job_id', type=str)
    parser.add_argument('--offline', action='store_true', default=False)  # wandb offline
    parser.add_argument('--test', action='store_true', default=False, help='Flag for testing')
    parser.add_argument('--knn', action='store_true', default=False, help='Flag for turning on the KNN eval')
    
    # SSL args
    parser.add_argument('--byol', action='store_true', default=False, help='Flag for training with BYOL')
    parser.add_argument('--simsiam', action='store_true', default=False, help='Flag for training with SimSiam')
    parser.add_argument('--dino', action='store_true', default=False, help='Flag for training with DINO')
    parser.add_argument('--dinov2', action='store_true', default=False, help='Flag for training with DINOv2')
    
    return parser.parse_args()


def update_params_from_args(params, args):
    # empty print
    print(f'')
    if args.warmup:
        params.optimization_params.default.scheduler.params.LinearWarmup.warmup_iters = args.warmup
        print(f'Changed warmup to: {args.warmup}')
        
    if args.epochs:
        params.training_params.epochs = args.epochs
        print(f'Changed epochs to: {params.training_params.epochs}')
        
    if args.num_workers:  # args.num_workers should be str so that "0" takes effect as well
        num_workers = int(args.num_workers)
        params.dataloader_params.trainloader.num_workers = num_workers
        params.dataloader_params.valloader.num_workers = num_workers
        params.dataloader_params.testloader.num_workers = num_workers
        print(f'Changed num_workers to: {num_workers}')
    
    if args.prefetch_factor:
        if args.prefetch_factor == 'None':
            prefetch_factor = None
            params.dataloader_params.trainloader.persistent_workers = False  # requires num_workers > 0
            params.dataloader_params.valloader.persistent_workers = False
            params.dataloader_params.testloader.persistent_workers = False
        else:
            prefetch_factor = int(args.prefetch_factor)
        params.dataloader_params.trainloader.prefetch_factor = prefetch_factor
        params.dataloader_params.valloader.prefetch_factor = prefetch_factor
        params.dataloader_params.testloader.prefetch_factor = prefetch_factor
        print(f'Changed prefetch_factor to: {prefetch_factor}')
    
    if args.pretrained_path:
        params.transfer_learning_params.pretrained_path = args.pretrained_path
        print(f'Using pretrained_path for initialization: {args.pretrained_path}')

    if args.lr:
        params.optimization_params.default.optimizer.params.lr = args.lr
        print(f'Changed lr to: {args.lr}')
    
    if args.wd is not None:  # can be 0
        params.optimization_params.default.optimizer.params.weight_decay = args.wd
        print(f'Changed weight decay to: {args.wd}')
    
    if args.dpr is not None:  # can be 0
        params.model_params.transformers_params.drop_path_rate = args.dpr
        print(f'Changed drop_path_rate to: {args.dpr}')
    
    if args.dr is not None:  # can be 0
        params.model_params.transformers_params.drop_rate = args.dr
        print(f'Changed drop_rate to: {args.dr}')
    
    if args.adr is not None:  # can be 0
        params.model_params.transformers_params.attn_drop_rate = args.adr
        print(f'Changed attn_drop_rate to: {args.adr}')

    if args.gpu:
        prev_gpu = params.system_params.which_GPUs
        params.system_params.which_GPUs = args.gpu  # change the value in-place
        print('Changed GPU for this run from {} to \033[1m{}\033[0m'.format(prev_gpu, args.gpu))

    if args.model_name:
        prev_model_name = params.training_params.model_name
        params.training_params.model_name = args.model_name
        print('Changed model_name for this run from {} to \033[1m{}\033[0m'.format(prev_model_name, args.model_name))

    if args.save_dir:
        params['training_params']['save_dir'] = args.save_dir
        print('Changed save_dir to: "\033[1m{}\033[0m"'.format(args.save_dir))

    if args.batch_size:
        for loader in ['trainloader', 'valloader', 'testloader']:
            params['dataloader_params'][loader]['batch_size'] = args.batch_size
            print('Changed \033[1m{}\033[0m batch_size to: \033[1m{}\033[0m'.format(loader, args.batch_size))

    if args.val_every is not None:
        params['training_params']['val_every'] = args.val_every
        print('Changed val_every to: \033[1m{}\033[0m'.format(args.val_every))
    
    if args.log_every is not None:
        params['training_params']['log_every'] = args.log_every
        print('Changed log_every to: \033[1m{}\033[0m'.format(args.log_every))
    
    if args.job_id is not None:
        params['training_params']['job_id'] = args.job_id
        print(f'Set job_id to: {args.job_id}')

    if args.mixed_precision:
        params.training_params.use_mixed_precision = True
        print(f'Explicitly set use_mixed_precision to: \033[1m{True}\033[0m')
    
    if args.knn:
        assert args.test, "args --test --knn should be used together"
        params['dataloader_params']['trainloader']['shuffle'] = False
        params['dataloader_params']['valloader']['shuffle'] = False
        params['dataloader_params']['testloader']['shuffle'] = False        
        params['training_params']['knn_eval'] = True
        params['model_params']['freeze_backbone'] = True
        print(f'Changed knn_eval and freeze_backbone to: True')
    
    # last empty print
    print(f'')

def main(parameters, args):
    # check if SSL training
    assert not args.byol * args.simsiam, "BYOL or SimSiam can be on but not both"
    use_momentum = True if args.byol else False 
    
    # instantiate wrapper with all its definitions   
    if args.byol or args.simsiam or args.dino:
        if args.dino:
            wrapper = DINOWrapper(parameters)
        else:
            wrapper = BYOLWrapper(parameters, use_momentum=use_momentum)
    else:
        wrapper = DefaultWrapper(parameters)
    wrapper.instantiate()
    
    # initialize wand logger
    if wrapper.is_rank0:
        log_params = wrapper.parameters.log_params    
        training_params = wrapper.parameters.training_params
        if wrapper.log_params['run_name'] == "DEFINED_BY_MODEL_NAME":
            log_params['run_name'] = training_params.model_name  
        if args.debug:
            os.environ['WANDB_MODE'] = 'dryrun'
        if not args.test:
            wandb_dir = parameters.training_params.save_dir.replace('checkpoints', 'wandb')
            mode = "offline" if args.offline else "online"
            print(f'Using wandb dir: {wandb_dir} -- mode: {mode}')
            wandb_dict=edict(
                project=log_params.project_name, 
                name=log_params.run_name, 
                config=wrapper.parameters,
                resume=True if training_params.restore_session else False,
                dir=wandb_dir
            )
            if mode == 'offline':
                wandb_dict.mode = 'offline'
            run = wandb.init(**wandb_dict)
    
    # define trainer 
    if args.byol or args.simsiam or args.dino or args.dinov2:
        if args.dino:
            trainer = DINOTrainer(wrapper)
        else:
            trainer = BYOLTrainer(wrapper, use_momentum)
    else:
        trainer = Trainer(wrapper)
        
    # set additional properties for the trainer (dry, debug)
    if args.debug:
        trainer.is_debug = True
    if args.dry:
        trainer.is_dry = True
    
    # do actual run (train/test)
    elif args.test or args.knn:
        assert args.pretrained_path  # should always be provided
        trainer.test(chpt_path=args.pretrained_path)
    else:
        trainer.train()
        if wrapper.is_supervised:
            trainer.test()
        # sync wandb once the run finishes
        if args.offline:
            wandb.finish()
            print(f'\n---- STARTING SYNCING at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ----\n')
            run_id = run.id
            pattern = f"{wandb_dir}/wandb/offline-run-*-{run_id}"
            run_dir = glob.glob(pattern)[0]
            #
            sync_command = f"wandb sync {run_dir}"
            print(f'Run id: {run_id} \nrun dir: {run_dir} \nRunning sys command: {sync_command}')
            os.system(sync_command)
            print(f'\n---- SYNCING DONE at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ----\n')


if __name__ == '__main__':
    # read and update params with args, set system props
    args = parse_arguments()
    print(f'\nUSING PARAMS FROM PATH: {os.path.abspath(args.params_path)}\n')
    if '_others' in args.params_path:
        parameters = edict(load_param_file(os.path.join(get_parent_path(args.params_path), '..', '__common__.yml')))
    else:
        parameters = edict(load_param_file(os.path.join(get_parent_path(args.params_path), '__common__.yml')))
    specific_params = edict(load_param_file(args.params_path))
    helpfuns.update_nested_values(parameters, specific_params)

    # update with args
    update_params_from_args(parameters, args)
    
    try:
        launch(main, (parameters, args))
    except Exception as e:       
        if dist.is_initialized():
            dist.destroy_process_group()            
        raise e
    finally:
        if dist.is_initialized():
            synchronize()         
            dist.destroy_process_group()            
