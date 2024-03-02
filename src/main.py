from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings
warnings.filterwarnings('ignore')

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import LambdaLR
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast

import tqdm
import pickle
import argparse
import random
import math
import os
import bisect

import dill

from sklearn.utils import shuffle
import numpy as np
import tqdm
from tqdm import tqdm
import time
import sys

import netclr
import augmentor 
from common import find_sizes 
from augmentor import Augmentor, getOutgoingPkts
from netclr import NetCLR
from backbone import DFNet, DFsimCLR

######################### SPECIFY THE PATH HERE #########################
DATA_ABSOLUTE_PATH = ''
LOG_ABSOLUTE_PATH = ''
#########################################################################

gpu_number = 1
use_cuda = torch.cuda.is_available()
device = torch.device(f"cuda:{gpu_number}" if use_cuda else "cpu")
kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
print(f"Device: {device}")


def parse_args():
    """
    Parse command line arguments

    Accepted arguments:
      (o)utput -- directory that contains model checkpoint
      (l)ogs_name -- file ane that saves results
      (b)atch_size    -- batch size for the epoch
      (n)um_epoches -- number of epochs
      (t)emperature -- this value is suggested by the original SimCLR paper
      fp16_precision     -- fp16_precision
      n_views     -- number of samplings

    Returns
    -------
    Namespace
        Argument namespace object

    """
    parser = argparse.ArgumentParser("Pretrain netCLR with customed features")

    # Required Arguments
    # directory containing feature files
    parser.add_argument("-o", "--output",
                        required=True,
                        type=str,
                        help="Directory which contains model checkpoints") #help = Help message for an argument

    parser.add_argument("-l", "--logs_name",
                        required=True,
                        type=str,
                        help="Directory file name that store logs of noise injection.")

    # Optional Arguments
    parser.add_argument("-b", "--batch_size",
                        type=int,
                        default=256,
                        help="the number of batch per epoch.")
    parser.add_argument("-n", "--num_epoches",
                        type=int,
                        default=100, # 401
                        help="The number of epoches.")
    parser.add_argument("-t", "--temperature",
                        type=float,
                        default=0.5,
                        help="this value is suggested by the original SimCLR paper")
    parser.add_argument("--fp16_precision",
                        type=bool,
                        default=True,
                        help="fp16 precision -> true or false.")
    parser.add_argument("--n_views",
                        type=int,
                        default=2,
                        help="The number of augments(sampling)")
    return parser.parse_args()


# Data Loaders
class TrainData(Dataset):
    def __init__(self, x_train, y_train, augmentor, n_views):
        self.x = x_train
        self.y = y_train
        self.augmentor = augmentor
        self.n_views = n_views
    
    def _aug(self, inp): #weak augmentation
        flip_idx = np.random.randint(0, 4999, 250)
        x_w = inp.copy()
        temp = x_w[flip_idx]
        x_w[flip_idx] = x_w[flip_idx+1]
        x_w[flip_idx+1] = temp 
        return x_w
    
    def __getitem__(self, index):
        return [self.augmentor.augment(self.x[index]) for i in range(self.n_views)], self.y[index]
    
    def __len__(self):
        return len(self.x)


def main(output_path, logs_name, batch_size = 256, num_epoches=100, temperature=0.5, fp16_precision=True, n_views = 2):
    """
    Run the command Starting from the directory : */rNetLR
    """
    # logs_name = 'nj_1_epoch_100.txt'

    # load dataset
    start_time = time.time() 

    logs_path = os.path.join(LOG_ABSOLUTE_PATH, logs_name) 
    
    print(f'model path : {output_path}\nlogs path : {logs_path}')
    print(f'batch: {batch_size}, epochs: {num_epoches}, temp: {temperature}, fp16: {fp16_precision}, n_views: {n_views}')
    
    ##### data path fixed ###############################################
    data_path = os.path.join(DATA_ABSOLUTE_PATH, 'cfg2_pretrain.npz') # 'cfg2_pretrain.npz'
    data = np.load(data_path) 
    print(f'data files loaded : {data.files}')

    x_train = data['X_pretrain_inferior'] 
    y_train = data['y_pretrain_inferior']
    print('splitted to x and y dataset')

    #####################################################################

    num_classes = len(np.unique(y_train))
    print (f"Number of classes: {num_classes}") #100
    print (f'Train data shapes: {x_train.shape}, {y_train.shape}')# (50000, 5000), (50000,)

    end_time = time.time()
    print(f'Loading time was {(end_time - start_time) // 60:.0f} min {(end_time - start_time) % 60:.0f} secs')


    # NetAugment
    OUTCOMING_SIZE_CDF, outcoming_sizes, max_outcoming_size = getOutgoingPkts(x_train)

    # Running the Pre-training
    augmentor = Augmentor(logs_path, OUTCOMING_SIZE_CDF, outcoming_sizes, max_outcoming_size)

    # n_view = 2
    train_dataset = TrainData(x_train, y_train, augmentor, 2) ## #TrainData(x_train, y_train, augmentor, 2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    df = DFNet(out_dim=512)
    model = DFsimCLR(df, out_dim=128).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003) #, weight_decay = 1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

    netclr = NetCLR(model = model,
               optimizer = optimizer,
               scheduler = scheduler,
               fp16_precision = fp16_precision,
               device = device,
               temperature = temperature,
               n_views = n_views,
               num_epoches = num_epoches, # 401
               batch_size = batch_size,
               filename = output_path
               )

    print('train netCLR')
    netclr.train(train_loader)


if __name__ == "__main__":
    try:
        args = parse_args()
        main(
            output_path = args.output, 
            logs_name = args.logs_name,
            batch_size = args.batch_size,
            num_epoches = args.num_epoches, 
            temperature = args.temperature,
            fp16_precision = args.fp16_precision,
            n_views =args.n_views
        )
    except KeyboardInterrupt:
        sys.exit(-1)        
