import os
import argparse

ROOT_DIR = "./dgcn/" 

def get_arguments():
    r"""Define all the configurations of the project here.
    Run the program from the terminal by "--" the args.
    Returns : a list of parsed arguments. 
    """ 

    parser = argparse.ArgumentParser(description = 'DGCNet')

    parser.add_argument('--seed', type = int,
                       default = 42,
                       help = 're-produce the results with random seed')   
    parser.add_argument('--rgb', type = str2bool, 
                        default =  False)

    '''
    Params for I/O locations and dataloader.
    '''
    parser.add_argument('--input_loc', type = str,
    					default = ROOT_DIR, help = 'data files location')
    parser.add_argument('--data_list', type = str,
                        help = 'path to the file listing the images in the dataset.')
    parser.add_argument('--model_loc', type = str,
    					default = ROOT_DIR, help = 'path for models')
    parser.add_argument('--load_from', type = str,
                        help = 'restore models from')
    parser.add_argument('--input_size', type = int, default = 832)
    parser.add_argument('--batch_size', type = int, default = 3,
                        help = 'no of images in a batch')
    parser.add_argument('--batch_size_per_gpu', type = int, default = 1,
                        help = 'no of images per gpu')
    parser.add_argument('--scale', action = "store_true", default = True,
                        help = 'randomly scale images during training') 
    parser.add_argument('--mirror', action = "store_true", default = True,
                        help = 'randomly mirror images during training')
    parser.add_argument('--ignore_label', type = int, default = 255, 
                        help = 'index of label to ignore during training')
    parser.add_argument('--checkpoint_freq', type = int, default = 5000,
                        help = 'frequency of summaries and checkpoints.')
  
    '''
    Params for training.
    '''
    parser.add_argument('--gpu_idx', type = str, default = None,
                        help = 'GPU index to use')
    parser.add_argument('--total_gpus', type = int, default = 3,
                        help = 'total gpus available')
    parser.add_argument('--no_cuda', type = int, default = None,
                        help = 'if true, cuda is not used')
    parser.add_argument('--opt_type', type = str, default = 'sgd')
    parser.add_argument('--lr', type = float, default = 1e-2,
                        help = 'learning_rate for training w/ polynomial decay')
    parser.add_argument('--mom', type = float, default = 0.9,
                        help = 'momentum for the optim')              
    parser.add_argument('--power', type = float, default = 0.9,
                        help = 'decay parameter')
    parser.add_argument('--w_decay', type = float, default = 5e-4,
                        help = 'regularization param')
    parser.add_argument("--num_workers", type = int, default = 8) 
    parser.add_argument('--iters', type = int, default = 10000,
                        help = 'number of training steps.')
    '''
    Params for model.
    '''
    parser.add_argument('--arch', type = str, default = "",
                        help = 'network arch')
    parser.add_argument('--classes', type = int, default = 19,
                        help = 'no. of classes to predict.')
    
    '''
    Params for OHEM
    '''
    parser.add_argument('--ohem', type = str2bool, default = False,
                        help = 'use hard negative mining')
    parser.add_argument('-ohem_threshold', type = float, default = 0.7, 
                        help = 'choose the samples with correct probability under the threshold')           
    parser.add_argument('-ohem_keep', type = int, default = 1e5, 
                        help = 'choose the samples with correct probability under the threshold')
    
    '''
    Params for distributed training.
    '''
    parser.add_argument('--apex', action = 'store_true', default = False,
                        help = 'Use Nvidia Apex Library')
    parser.add_argument('--local_rank', type = int, default = 0,
                        help = 'param for apex module.')       
    args = parser.parse_args()

    return args




def str2bool(a):
    if a.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif a.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Expected a boolean value')
