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

# The backbone of NetCLR model is the Deep Fingerprinting neural network. 
class DFNet(nn.Module):
    """
    input :  (__, 5000)
    output : (5000, )
    """
    def __init__(self, out_dim):
        super(DFNet, self).__init__()
        kernel_size = 8
        channels = [1, 32, 64, 128, 256]
        conv_stride = 1
        pool_stride = 4
        pool_size = 8
        
        self.conv1 = nn.Conv1d(1, 32, kernel_size, stride = conv_stride)
        self.conv1_1 = nn.Conv1d(32, 32, kernel_size, stride = conv_stride)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size, stride = conv_stride)
        self.conv2_2 = nn.Conv1d(64, 64, kernel_size, stride = conv_stride)
       
        self.conv3 = nn.Conv1d(64, 128, kernel_size, stride = conv_stride)
        self.conv3_3 = nn.Conv1d(128, 128, kernel_size, stride = conv_stride)
       
        self.conv4 = nn.Conv1d(128, 256, kernel_size, stride = conv_stride)
        self.conv4_4 = nn.Conv1d(256, 256, kernel_size, stride = conv_stride)
       
        
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.batch_norm4 = nn.BatchNorm1d(256)
        
        self.max_pool_1 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        self.max_pool_2 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        self.max_pool_3 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        self.max_pool_4 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.1)
        self.dropout4 = nn.Dropout(p=0.1)

        
        self.fc = nn.Linear(5120, out_dim)

        
    def weight_init(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
#                 m.weight.data.xavier_uniform_()
                # print (n)
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.zero_()
            
        
    def forward(self, inp):
        x = inp
        # ==== first block ====
        x = F.pad(x, (3,4))
        x = F.elu((self.conv1(x)))
        x = F.pad(x, (3,4))
        x = F.elu(self.batch_norm1(self.conv1_1(x)))
#         x = F.elu(self.conv1_1(x))
        x = F.pad(x, (3, 4))
        x = self.max_pool_1(x)
        x = self.dropout1(x)
        
        # ==== second block ====
        x = F.pad(x, (3,4))
        x = F.relu((self.conv2(x)))
        x = F.pad(x, (3,4))
        x = F.relu(self.batch_norm2(self.conv2_2(x)))
#         x = F.relu(self.conv2_2(x))
        x = F.pad(x, (3,4))
        x = self.max_pool_2(x)
        x = self.dropout2(x)
        
        # ==== third block ====
        x = F.pad(x, (3,4))
        x = F.relu((self.conv3(x)))
        x = F.pad(x, (3,4))
        x = F.relu(self.batch_norm3(self.conv3_3(x)))
#         x = F.relu(self.conv3_3(x))
        x = F.pad(x, (3,4))
        x = self.max_pool_3(x)
        x = self.dropout3(x)
        
        # ==== fourth block ====
        x = F.pad(x, (3,4))
        x = F.relu((self.conv4(x)))
        x = F.pad(x, (3,4))
        x = F.relu(self.batch_norm4(self.conv4_4(x)))
#         x = F.relu(self.conv4_4(x))
        x = F.pad(x, (3,4))
        x = self.max_pool_4(x)
        x = self.dropout4(x)

                
        x = x.view(x.size(0), -1)
        
#         x = self.projection(x)

        x = self.fc(x)
                
        return x    

# Projection Head
class DFsimCLR(nn.Module):
    """
    (5000, ) -> (512, ) -> fc, ReLU, BN, fc -> (128, )
    """
    def __init__(self, df, out_dim):
        super(DFsimCLR, self).__init__()
        
        self.backbone = df
        self.backbone.weight_init()
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.BatchNorm1d(dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, out_dim)
        )
        
    def forward(self, inp):
        out = self.backbone(inp)
        return out
