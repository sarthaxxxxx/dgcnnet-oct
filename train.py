import os
import csv 
import cv2 
import sys
import random
import timeit
import argparse
import warnings
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
import torch.optim as opt 
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from configs.cfg import get_arguments, str2bool
import src.models as model
from utils.train_tools import str2bool, create_optimizer, adjust_lr, reduce_tensor
from utils.dataset import Cityscapes
from utils.losses import CriterionDSN, CriterionOhemDSN

    
start = timeit.default_timer()

def main():

    cfg = get_arguments()
    '''
    For reproducibility and controlling randomness. 
    '''
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        
    if cfg.local_rank == 0:
        if not os.path.exists(cfg.model_loc):
            os.makedirs(cfg.model_loc)

    cfg.world_size = 1

    main_worker(cfg)


def main_worker(cfg):

    if cfg.gpu_idx is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_idx

    if cfg.rgb:
        mean = np.array((0.485, 0.456, 0.406), dtype = np.float32)
        var = np.array((0.229, 0.224, 0.225), dtype = np.float32)
    else:
        mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype = np.float32)
        var = np.array((1, 1, 1), dtype = np.float32)


    if 'WORLD_SIZE' in os.environ and cfg.apex:
        cfg.apex = int(os.environ['WORLD_SIZE']) > 1
        cfg.world_size = int(os.environ['WORLD_SIZE'])
        print(f"Total world size : {int(os.environ['WORLD_SIZE'])}")


    dgcn = model.__dict__[cfg.arch](classes = cfg.classes)

    if cfg.load_from is not None:
        saved_state_dict = torch.load(cfg.load_from, map_location = torch.device('cpu'))
        new_parameters = dgcn.state_dict().copy()
        for i in saved_state_dict:
            i_parts = i.split('.')
            if not i_parts[0] == 'fc':
                new_parameters['.'.join(i_parts[0:])] = saved_state_dict[i]
        dgcn.load_state_dict(new_parameters, strict = False)
    else:
        print('*'*40)
        print('Training from scratch')
        print('*'*40)

    torch.cuda.set_device(cfg.local_rank)
    torch.distributed.init_process_group(backend = 'nccl', init_method = 'env://')

    dgcn.cuda()
    model = DistributedDataParallel(dgcn)
    model = apex.parallel.convert_syncbn_model(dgcn)
    model.train().float()

    optimizer = create_optimizer(cfg, dgcn)
    optimizer.zero_grad()

    if cfg.ohem:
        loss_criterion = CriterionOhemDSN(thresh = cfg.ohem_threshold,
                                         min_kept = cfg.ohem_keep)   
    else:
        loss_criterion = CriterionDSN()

    loss_criterion.cuda()

    cudnn.benchmark = True

    if cfg.world_size == 1:
        print(dgcn)


    batch_size = cfg.total_gpus * cfg.batch_size_per_gpu
    max_iters = cfg.iters * cfg.batch_size_per_gpu


    dataset = Cityscapes(listdir = cfg.data_list,
                         mean = mean,
                         var = var,
                         crop_size = (cfg.input_size, cfg.input_size),
                         scale = cfg.scale,
                         mirror = cfg.mirror,
                         ignore_label = cfg.ignore_label,
                         rgb = cfg.rgb,
                         max_iters = max_iters)

    train_loader = data.DataLoader(dataset,
                                   batch_size = cfg.batch_size_per_gpu,
                                   shuffle = True,
                                   num_workers = cfg.num_workers,
                                   pin_memory = True)

    if len(train_loader) is not None:
        print('*'*40)
        print('Loaded dgcn and train dataloader succesfully')
        print('*'*40)

    torch.cuda.empty_cache()

    trainer(train_loader, optimizer, loss_criterion, cfg, dgcn, model) #train the model and get preds

    end = timeit.default_timer()

    if cfg.local_rank == 0:
        print(f'Training duration : {end - start} seconds')
        print('Saving final best model')
        torch.save(dgcn.state_dict(), os.path.join(cfg.model_loc, str(cfg.arch) + '_final' + '.pth'))


def trainer(train_loader, optim, loss_criterion, cfg, dgcn, model):

    for idx, (images, gt) in tqdm(enumerate(train_loader)):
        images = images.cuda()
        gt = gt.cuda().long()
        optim.zero_grad()
        lr = adjust_lr(optim, cfg, idx, len(train_loader))
        _preds = model(images)
        loss = loss_criterion(_preds, gt)
        loss.backward()
        optim.step()        

        reduce_loss = reduce_tensor(loss, world_size = cfg.total_gpus)

        if cfg.local_rank == 0:
            print('*'*40)
            print(f"Iteration : {idx}/{len(train_loader)}, lr : {lr}, loss : {reduce_loss.data.cpu().numpy()}")
            print('*'*40)

            if idx % cfg.checkpoint_freq == 0:
                print(f'Saving model at iteration: {idx}')
                torch.save(dgcn.state_dict(), os.path.join(cfg.model_loc, str(cfg.arch) + str(idx) + '.pth')) 

    
if __name__ == '__main__':
    '''
    For multiprocessing distributed training.
    '''
    try:
        os.system('git clone https://github.com/NVIDIA/apex; cd apex; pip install -v --no-cache-dir' + 
                 ' --global-option="--cpp_ext" --global-option="--cuda_ext" ./')  #install nvidia-apex
        os.system('rm -rf apex/.git')
        import apex
        from apex import amp
        from apex.apex.parallel import DistributedDataParallel, SyncBatchNorm
    except ImportError:
        raise ImportError ('Apex not found for distributed training!!!')

    main()


    
