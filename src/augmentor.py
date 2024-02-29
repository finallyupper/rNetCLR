from __future__ import absolute_import ##__future__ 사용하면 최신 버전 불러올 수 있음.
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
from common import find_sizes 

def getOutgoingPkts(x_train):
    outcoming_sizes = []
    x_random = x_train[np.random.choice(range(len(x_train)), size=1000, replace=False)]  #1000 traces random selected

    for x in x_random:
        sizes = find_sizes(x) ##
        outcoming_sizes += [size for size in sizes if size > 0] 
    
    max_outcoming_size = max(outcoming_sizes)
    print(len(outcoming_sizes), min(outcoming_sizes), max(outcoming_sizes) ) # (199148, 1, 126) ###check

    # Empirical Distribution of Outgoing Packets
    max_outcoming_size = int(max_outcoming_size)
    count, bins = np.histogram(sizes, bins=(max_outcoming_size) - 1)
    PDF = count/np.sum(count)
    OUTCOMING_SIZE_CDF = np.zeros_like(bins)
    OUTCOMING_SIZE_CDF[1:] = np.cumsum(PDF)
    print('Calculated OUTCOMING_SIZE_CDF !')

    return OUTCOMING_SIZE_CDF, outcoming_sizes, max_outcoming_size


class Augmentor():
    def __init__(self, logs_path, OUTCOMING_SIZE_CDF, outcoming_sizes, max_outcoming_size, method):
        methods = [
            'noise injection',
            'add outcoming packets'
        ]
        
        # save logs for noise injection
        self.logs_path = logs_path 
        # 2. add outcoming packets
        self.OUTCOMING_SIZE_CDF = OUTCOMING_SIZE_CDF
        self.outcoming_sizes = outcoming_sizes
        self.max_outcoming_size = max_outcoming_size
        self.add_outcoming_rate = 0.3
        self.outcoming_size = list(range(max_outcoming_size))
        self.method = method

        
    # def find_times_sizes(self, t, s):
    #     """
    #     input x shape = (times, sizes) -> size info only

    #     parameters
    #     ----
    #     x : nd array

    #     returns
    #     ----
    #     sizes : int list
    #     """
    #     times = x[:, 0]
    #     sizes = x[:, 1]

    #     return list(t), list(s) #list(times), list(sizes)

    # Inserting Outcoming packets
    def add_outcoming_packet(self, packets):
        
        out = []
        
        i = 0
        num_cells = 0
        while i < len(packets) and num_cells < 38: #updated 2/25 ####조작시 처음 20cell은 냅둠 (=for protocol 초기화, handshake// 웹사이트 구별용)
            num_cells += abs(packets[i])
            out.append(packets[i])
            i += 1  ## = len(out)
            
        
        for size in packets[i:]:
            if size > -10 :
                out.append(size)
                continue
            
            prob = random.random()
            
            if prob < self.add_outcoming_rate:
                
                index = len(self.outcoming_sizes)
                while index >= len(self.outcoming_sizes):
                    outcoming_size_prob = random.random()
                    index = bisect.bisect_left(self.OUTCOMING_SIZE_CDF, outcoming_size_prob) # distribution for sampling
                    
                outcoming_sizes = self.outcoming_sizes[index] # get random size
                divide_place = random.randint(3, abs(size) - 3) # random position
                
                out += [-divide_place, outcoming_sizes, -(abs(size) - divide_place)]
                
            else:
                out.append(size)
                
        return out
                
    ############################### ADD YOUR CUSTOMED MANIPULATIONS HERE###################################
    def myfunction(self):
        pass 

    ########################################################################################################
    def create_trace_from_sizes(self, sizes):
        """
        input: ndarray (1dim)
        output: ndarray (5000,)
        """
        _tmp = np.array((0.0, 0.0)).reshape((1, 2))
        if len(sizes) < 5000: # zero padding
            amount = 5000 - len(sizes)
            tmp = np.repeat(_tmp, amount, axis = 0)
            out = np.concatenate((sizes, tmp), axis = 0)
        else:
            out = sizes
            
        return out[:5000] # truncate if >5000
    
    def shift(self, x):
        pad = np.random.randint(0, 2, size = (self.shift_param, ))
        pad = 2*pad-1
        zpad = np.zeros_like(pad)
        
        shift_val = np.random.randint(-self.shift_param, self.shift_param+1, 1)[0]
        shifted = np.concatenate((x, zpad, pad), axis=-1)
        shifted = np.roll(shifted, shift_val, axis=-1) #
        shifted = shifted[:5000] # truncate if >5000
        
        return shifted
        
    
    def augment(self, trace):
        
        mapping = {
            0 : self.add_outcoming_packet,
            1 : self.myfunction
        }
        #print('trace shape' , trace.shape) # trace shape (5000, 2)

        #times, sizes = self.find_times_sizes(trace) 
        times = trace[:, 0]
        sizes = trace[:, 1]     

        ## changed to single manipulation available condition
        aug_method = mapping[self.method] #mapping[random.randint(0, len(mapping)-1)] #Randomly pick from manipulations(1,2,3)
        
        if aug_method == self.add_outcoming_packet:
            augmented_sizes = self.add_outcoming_packet(sizes)  

        augmented_trace = self.create_trace_from_sizes(augmented_sizes)
        
        shifted_trace = self.shift(augmented_trace)
        #print(f'shape of shifted trace : {shifted_trace.shape}') # (5000, )
        return shifted_trace