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
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.autograd import Variable
import torch.backends.cudnn as cudnn


from configs.cfg import get_arguments, str2bool
import src.models as model
from utils.train_tools import str2bool, create_optimizer, adjust_lr, reduce_tensor
from utils.dataset import Cityscapes
from utils.losses import CriterionDSN, CriterionOhemDSN


def main():

    cfg = get_arguments()
    
    if not os.path.exists(cfg.model_loc):
        os.makedirs(cfg.model_loc)

    cfg.world_size = cfg.total_gpus # world_size = gpus_per_node * nodes (1)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '6006'
    mp.spawn(main_worker, 
            args = (cfg, ),
            nprocs = cfg.total_gpus)
 

def main_worker(gpu, cfg):
    
    rank = cfg.local_rank * cfg.total_gpus + gpu

    dist.init_process_group(backend = 'nccl', 
                            init_method = 'env://',
                            world_size = cfg.world_size,
                            rank = rank)

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)   

    if cfg.rgb:
        mean = np.array((0.485, 0.456, 0.406), dtype = np.float32)
        var = np.array((0.229, 0.224, 0.225), dtype = np.float32)
    else:
        mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype = np.float32)
        var = np.array((1, 1, 1), dtype = np.float32)

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
        print('Training from scratch!')
        print('*'*40)

    torch.cuda.set_device(gpu)
    dgcn.cuda(gpu)
    model = nn.parallel.DistributedDataParallel(dgcn, 
                                                device_ids = [gpu])

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

    dataset = Cityscapes(listdir = cfg.data_list,
                         mean = mean,
                         var = var,
                         crop_size = (cfg.input_size, cfg.input_size),
                         scale = cfg.scale,
                         mirror = cfg.mirror,
                         ignore_label = cfg.ignore_label,
                         rgb = cfg.rgb)

    train_sampler = data.DistributedSampler(dataset,
                                            num_replicas = cfg.world_size,
                                            rank = rank)

    train_loader = data.DataLoader(dataset,
                                   batch_size = cfg.batch_size,
                                   shuffle = True,
                                   num_workers = cfg.num_workers,
                                   pin_memory = True,
                                   sampler = train_sampler)

    if len(train_loader) is not None:
        print('*'*40)
        print('Loaded dgcn and train dataloader succesfully')
        print('*'*40)

    torch.cuda.empty_cache()

    start = timeit.default_timer()

    for epoch in range(cfg.epochs):
        for idx, (images, gt) in tqdm(enumerate(train_loader)):
            images = images.cuda(non_blocking = True)
            gt = gt.cuda(non_blocking = True).long()
            # forward pass
            lr = adjust_lr(optimizer, cfg, idx, len(train_loader))
            _preds = model(images)
            #backward pass and optimize
            optimizer.zero_grad()
            loss = loss_criterion(_preds, gt)
            loss.backward()
            optimizer.step()        

            #reduce_loss = reduce_tensor(loss, world_size = cfg.world_size)

            if cfg.gpu == 0:
                print(f"Epoch: [{epoch + 1}/{cfg.epochs}], lr : {lr}, loss : {loss.item()}")

                if epoch % cfg.checkpoint_freq == 0:
                    print(f'Saving model at epoch: {epoch + 1}')
                    torch.save(dgcn.state_dict(), os.path.join(cfg.model_loc, str(cfg.arch) + str(idx) + '.pth')) 

    end = timeit.default_timer()

    if cfg.gpu == 0:
        print(f'Training duration : {end - start} seconds')
        print('Saving final best model')
        torch.save(dgcn.state_dict(), os.path.join(cfg.model_loc, str(cfg.arch) + '_final' + '.pth'))

    
if __name__ == '__main__':
    main()
