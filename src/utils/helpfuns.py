import os
import sys
import json
import pickle
import shutil
import random
import inspect
import warnings
import subprocess
import numpy as np
import pandas as pd
import glob
from PIL import Image
from tqdm import tqdm
from copy import deepcopy
# from pprint import pprint
from tabulate import tabulate
import matplotlib.pylab as plt
from collections import OrderedDict
from easydict import EasyDict as edict
import yaml
from utils.colors import *
from pprint import pformat


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
def upgrade(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])
    

def check_parent_dir(path):
    parentdir = get_parent_path(path)
    if not os.path.exists(parentdir):
        os.makedirs(parentdir, exist_ok=True)  # exist_ok: does not raise error if folder already exists        


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)  # exist_ok: does not raise error if folder already exists        


def load_param_file(paramfile):
    if paramfile.endswith('.json'):
        return load_json(paramfile)
    elif paramfile.endswith('.yml') or paramfile.endswith('.yaml'):
        return load_yaml(paramfile)
    else:
        raise NotImplementedError

def get_parent_path(path):
    # return os.path.abspath(os.path.join(path, os.pardir))
    return os.path.split(path)[0]
    
def save_json(data, fname):
    fname = os.path.abspath(fname)
    if not fname.endswith('.json'):
        fname += '.json'
    with open(fname, 'w') as wfile:  
        json.dump(data, wfile, indent=4)


def save_yml(data, fname):
    with open(fname, 'w') as wfile:
        yaml.dump(data, wfile, default_flow_style=False, sort_keys=False, indent=4)

        
def load_json(fname):
    fname = os.path.abspath(fname)
    with open(fname, "r") as rfile:
        data = json.load(rfile)
    return data

def save_pickle(data, fname):
    fname = os.path.abspath(fname)
    if not fname.endswith('.pickle'):
        fname += '.pickle'    
    with open(fname, 'wb') as wfile:
        pickle.dump(data, wfile, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_pickle(fname):
    fname = os.path.abspath(fname)
    with open(fname, 'rb') as rfile:
        data = pickle.load(rfile)
    return data      

def get_saved_model_path(checkpoint_name):
    path = os.path.join(os.getcwd(), 'checkpoints')
    if not os.path.exists(path):
        raise IOError("Checkpoint path {} does not exist".format(path))
    else:
        return os.path.join(path, checkpoint_name) 
    
def load_params(args):
    if args.checkpoint:
        import torch
        checkpoint_path = get_saved_model_path(args.checkpoint)
        checkpoint = torch.load(checkpoint_path)
        checkpoint['parameters']['training_params']['model_name'] = args.checkpoint
        return checkpoint['parameters']    
    elif args.params_path:
        return load_json(args.params_path)

    raise IOError("Please define the training paramenters")
        
def isnan(x):
    return x != x

def iszero(x):
    return x == 0    


def update_nested_values(_dict, _target, pname=[]):
    for key, value in _target.items():
        pname_copy = deepcopy(pname)  # make a deep copy to make sure different copies of list are passed to separate recursion branches
        pname_copy.append(key)
        # print(f'****** Branch for key: "{green(key)}" -- pname_copy: {green(str(pname_copy))}')
        if isinstance(_target[key], dict):
            # recurse until the value is not a dict itself
            if key not in _dict:
                _dict[key] = _target[key]
                print(f"Nested values: {blue('ADDED')} new key to _dict: key: '{key}' \nhierarchy: \"{'.'.join(pname_copy)}\" with value: {pformat(_dict[key], sort_dicts=False)}")
                continue  # skip traversing this newly added sub-dict
                # return  # this kills the siblings in the same recursion level!!!
            update_nested_values(_dict[key], _target[key], pname_copy)
        else:
            if key not in _dict:
                _dict[key] = _target[key]
                print(f"Nested values: {blue('ADDED')} new key to _dict: key: '{key}' \nhierarchy: \"{'.'.join(pname_copy)}\" with value: {_dict[key]}")
            else:
                _prevval = _dict[key]
                _dict[key] = _target[key]
                print(f"Nested values: {byello('UPDATED')} \"{'.'.join(pname_copy)}\" form: {_prevval} to: {_dict[key]}")

def load_yaml(filepath):
    with open(filepath, 'r') as file:
        params = yaml.safe_load(file)
    return params


def files_with_suffix(directory, suffix, pure=False):
    # files = [os.path.abspath(path) for path in glob.glob(f'{directory}/**/*{suffix}', recursive=True)]  # full paths
    files = [os.path.abspath(path) for path in glob.glob(os.path.join(directory, '**', f'*{suffix}'), recursive=True)]  # full paths
    if pure:
        files = [os.path.split(file)[-1] for file in files]
    return files


def write_list_to_file(filepath, the_list):
    with open(filepath, 'w') as handle:
        for item in the_list:
            handle.write(f'{item}\n')


def read_file_to_list(filepath, type_fn=None):
    with open(filepath, 'r') as handle:
        lines = handle.read().splitlines()
    if type_fn is not None:
        lines = [type_fn(line) for line in lines]
    return lines


def count_trainable_params(model):
    # for name, p in model.named_parameters():
    #     if p.requires_grad:
    #         print(f'{name} -> {p.numel():,}')
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return count


def count_total_params(model):
    count = sum(p.numel() for p in model.parameters())
    return count

