import os
import argparse

import torch 
import torch.optim as optim
import torch.distributed as dist


def str2bool(a):
    if a.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif a.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Expected a boolean value')


def create_optimizer(cfg, model):

    model_params = [params for params in model.parameters() if params.requires_grad]
    if cfg.opt_type == 'adam':
        optimizer = torch.optim.Adam(model_params, lr = cfg.lr, 
                                     weight_decay = cfg.w_decay)
    elif cfg.opt_type == 'sgd':
        optimizer = torch.optim.SGD(model_params, lr = cfg.lr, 
                                    momentum=cfg.mom, weight_decay=cfg.w_decay)
    else:
        assert False, "Unknown type of optimizer"

    return optimizer


def adjust_lr(optim, cfg, idx, len_trainer):
    lr = lr_poly(cfg.lr, idx, len_trainer, cfg.power)
    optim.param_groups[0]['lr'] = lr
    return lr


def lr_poly(base_lr, idx, max_iter, power):
    return base_lr * (1 - (float(idx) / max_iter)) ** power


def reduce_tensor(tensor, op = dist.ReduceOp.SUM, world_size = 1):
    tensor = tensor.clone()
    dist.all_reduce(tensor, op)
    tensor.div_(world_size)
    return tensor