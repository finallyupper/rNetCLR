from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings
warnings.filterwarnings('ignore')
import numpy as np

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
# from torchvision import datasets, transforms
import tqdm
import pickle
import argparse
from torch.cuda.amp import GradScaler, autocast

import random
import sys
import os
import collections


def train(model, device, train_loader, optimizer, epoch, filepath):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(data.size(0), 1, data.size(1)).float().to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        #print (output.size())
        
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx%100 == 0:
        #    print ("Loss: {:0.6f}".format(loss.item()))
            with open(filepath, 'a') as f:
                f.write(f'Epoch: {epoch}\t')
                f.write("Loss: {:0.6f}\n".format(loss.item()))
    
def test(model, device, loader):
    model.eval()
    correct = 0
    temp = 0
    with torch.no_grad():
        for data, target in loader:
            data = data.view(data.size(0), 1, data.size(1)).float().to(device)
            target = target.to(device)
            
            output = model(data)
            output = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).float().sum().item()
    return correct / len(loader.dataset)