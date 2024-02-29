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
from tqdm import tqdm
import pickle
import argparse
from torch.cuda.amp import GradScaler, autocast

import random
import sys
import os
import collections
import matplotlib.pyplot as plt
from pathos.multiprocessing import cpu_count
from pathos.multiprocessing import ProcessPool as Pool
import multiprocessing as mp
 
from backbone import DFNet, DFsimCLR
from cw_train import train, test 
#from typing_extensions import Literal
from typing import Union #, Literal  

# GPU Allocation
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:3" if use_cuda else "cpu") ##
kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
print (f'Device: {device}')

############################### edit here ###############################################
CFG = {
    'dataset' : 'CFG1', # CFG2
    'models' : ['addo_2_e__80.pth.tar',  'nj_3_e__80.pth.tar'],
    'trials': [1, 1],
    'base_path' : '/home/yoojinoh/winter_internship/Realistic-Website-Fingerprinting-By-Augmenting-Network-Traces/artifacts/src/rNetCLR/models/',
    'log_path' : '/home/yoojinoh/winter_internship/Realistic-Website-Fingerprinting-By-Augmenting-Network-Traces/artifacts/src/rNetCLR/finetuning/log',
    'output_path' : '/home/yoojinoh/winter_internship/Realistic-Website-Fingerprinting-By-Augmenting-Network-Traces/artifacts/src/rNetCLR/finetuning/results/',
    'N' : 5,
    'batch_size' : 32,
    'epoches': 100,
}
#########################################################################################

# This function randomly samples N traces per website
def sample_traces(x, y, N, num_classes):
    train_index = []
    
    for c in range(num_classes):
        idx = np.where(y == c)[0]
        idx = np.random.choice(idx, min(N, len(idx)), False)
        train_index.extend(idx)
        
    train_index = np.array(train_index)
    np.random.shuffle(train_index)
    
    x_train = x[train_index]
    y_train = y[train_index]
    
    return x_train, y_train

# Data Loader
class Data(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)


def load_checkpoint(model_path, num_classes):

    model = DFNet(out_dim=num_classes).to(device)

    checkpoint = torch.load(model_path)

    for k in list(checkpoint.keys()):
        if k.startswith('backbone.'):
            if k.startswith('backbone') and not k.startswith('backbone.fc'):
          # remove prefix
                checkpoint[k[len("backbone."):]] = checkpoint[k]
        del checkpoint[k]

    log = model.load_state_dict(checkpoint, strict=False)
    assert log.missing_keys == ['fc.weight', 'fc.bias']
    
    return model

class FineTuner:
    def __init__(self, dataset:str, 
                 base_path:str, 
                 model_path:str, 
                 output_path:str, 
                 log_path:str,
                 trial:int = 1, 
                 batch_size:int = 32,          
                 epoches:int = 100,
                 N:int = 5,)->None:
        #self.device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        self.x_train_total = None 
        self.y_train_total = None 
        self.num_classes = 69 
        self.dataset = dataset 
        self.base_path = base_path 
        self.model_path = model_path
        self.output_path = output_path 
        self.log_path = log_path
        self.trial = trial 
        self.batch_size = batch_size 
        self.epoches = epoches 
        self.N = N

    def load_dataset(self, DATASET:str):
        if DATASET == 'CFG1':    
            data_path = '/scratch/DA/dataset/tcp5/cfg1_fine_tuning_data.npz' # cfg1
            data = np.load(data_path) #pickle.load(open(f'{data_path}', 'rb'))
            
        elif DATASET == 'CFG2':
            data_path = '/scratch/DA/dataset/tcp5/cfg2_fine_tuning_data.npz' # cfg2
            data = np.load(data_path) #pickle.load(open(f'{data_path}', 'rb'))
        print(f'Successfully Loaded dataset for {DATASET}.')
      
        self.x_train_total = data['x_train'][:, :, 1] # size info only .. (no time info)
        self.y_train_total = data['y_train']
        x_test_sup = data['x_test_fast'][:, :, 1]
        y_test_sup = data['y_test_fast']
        x_test_inf = data ['x_test_slow'][:, :, 1]
        y_test_inf = data['y_test_slow']
        print('Successfully allocated data to variables.')

        self.num_classes = len(np.unique(self.y_train_total))
        print ("Number of classes:", self.num_classes) # Number of classes: 69
        print (f'Data shapes: {self.x_train_total.shape}, {x_test_sup.shape}, {x_test_inf.shape}') # Data shapes: (122820, 5000, 2), (3450, 5000, 2), (3450, 5000, 2)

        # initiating test data loaders
        test_dataset_inf = Data(x_test_inf, y_test_inf)
        test_loader_inf = DataLoader(test_dataset_inf, batch_size=self.batch_size, drop_last=True)

        test_dataset_sup = Data(x_test_sup, y_test_sup)
        test_loader_sup = DataLoader(test_dataset_sup, batch_size=self.batch_size, drop_last=True)
        print('initiated test data loaders')

        return test_loader_inf, test_loader_sup
    
    def _finetuning(self, 
                       model_path:str,
                        trial:int, 
                       test_loader_inf:DataLoader, 
                       test_loader_sup:DataLoader)->None:
        t = '_' + str(trial) 

        folder_path = self.output_path+ model_path.split('/')[-1].split('.')[0]  + t
        
        if os.path.exists(folder_path):
            if os.listdir(folder_path):
                raise Exception(f"The folder {folder_path.split('/')[-1]} already exists.")
        else:
            os.makedirs(folder_path)

        save_filepath = self.log_path +  '/ft_' + model_path.split('/')[-1].split('.')[0] + t + '.txt'   
        _save_for_graph_path = model_path.split('/')[-1].split('.')[0] + '_graph' + t + '.pickle'
        save_for_graph_path = os.path.join(folder_path, _save_for_graph_path)
        
        # Running for 5 times
        accuracies_inf = []
        accuracies_sup = []

        with open(save_filepath, 'a') as f:
            f.write(f'Running for {self.N} times\n')
        
        # save results
        trial = []
        x = []; sup_y = []; inf_y = []

        for cnt in range(self.N): 
            print(f'[ # of Runnings = {cnt} / {self.N} ]')
            with open(save_filepath, 'a') as f:
                f.write(f'============== Runned : {cnt + 1}/{self.N} ===============\n')
            
            trial.append(cnt)
            _x = []
            _sup_y = []
            _inf_y = []
            
            x_train, y_train = sample_traces(self.x_train_total, self.y_train_total, self.N, self.num_classes)
            
            #print ("Input size:", x_train.shape, y_train.shape) 
            
            train_dataset = Data(x_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
            
            model = load_checkpoint(model_path=model_path, num_classes = self.num_classes)
            optimizer = optim.Adam(model.parameters(), lr=0.0001)
            
            
            best_acc_inf = 0
            best_acc_sup = 0

            for epoch in range(self.epoches):
            #    print ('Epoch: ', epoch)

                train(model, device, train_loader, optimizer, epoch, save_filepath)
                
                acc_inf = test(model, device, test_loader_inf)
                acc_sup = test(model, device, test_loader_sup)
                
                best_acc_inf = max(best_acc_inf, acc_inf)
                best_acc_sup = max(best_acc_sup, acc_sup)
                
                with open(save_filepath, 'a') as f:
                    f.write(f"Accuracy on inferior dataset: {acc_inf*100:.2f}\n")
                    f.write(f"Accuracy on superior dataset: {acc_sup*100:.2f}\n")

                _x.append(epoch)
                _sup_y.append(acc_sup*100)
                _inf_y.append(acc_inf*100)

                # if epoch%10 == 0:
                #     #print (f"Accuracy on inferior dataset: {acc_inf*100:.2f}")
                #     #print (f"Accuracy on superior dataset: {acc_sup*100:.2f}")
                #     f.write(f'Epoch: {epoch}')
                #     f.write(f"Accuracy on inferior dataset: {acc_inf*100:.2f}")
                #     f.write(f"Accuracy on superior dataset: {acc_sup*100:.2f}")
        
            x.extend([_x])
            sup_y.extend([_sup_y])
            inf_y.extend([_inf_y])

            accuracies_inf.append(best_acc_inf)
            accuracies_sup.append(best_acc_sup)
            #print (f"Run {cnt + 1}> Accuracy on inferior & superior dataset: {best_acc_inf*100:.2f}, {best_acc_sup*100:.2f}")

            with open(save_filepath, 'a') as f:
                f.write('------------------------------------------------\n\n')
        # Ended for loop
            
        results = dict()
        results['N']=trial
        results['x'] = x
        results['sup_y'] = sup_y
        results['inf_y'] = inf_y 
        results['best_sup_y'] = accuracies_sup
        results['best_inf_y'] = accuracies_inf
        
        with open(save_for_graph_path, 'wb') as f2:
            pickle.dump(results, f2)
            print(f'Saved Results to pickle file as {save_for_graph_path}!')

        accuracies_inf = np.array(accuracies_inf)
        accuracies_sup = np.array(accuracies_sup)

        #print (f"Test accuracy on inferior traces: avg -> {np.mean(accuracies_inf)*100:.1f}, std -> {np.std(accuracies_inf)*100:.1f}")
        #print (f"Test accuracy on Superior traces: avg -> {np.mean(accuracies_sup)*100:.1f}, std -> {np.std(accuracies_sup)*100:.1f}")
        with open(save_filepath, 'a') as f:
            f.write(f"Test accuracy on inferior traces: avg -> {np.mean(accuracies_inf)*100:.1f}, std -> {np.std(accuracies_inf)*100:.1f}\n")
            f.write(f"Test accuracy on Superior traces: avg -> {np.mean(accuracies_sup)*100:.1f}, std -> {np.std(accuracies_sup)*100:.1f}\n\n")

    

    def finetuning(self):
        test_loader_inf, test_loader_sup = self.load_dataset(self.dataset)
        self._finetuning(self.model_path, self.trial, test_loader_inf , test_loader_sup)

def run_finetuner(dataset:str, 
                 base_path:str, 
                 model_path:str, 
                 output_path:str, 
                 log_path:str,
                 trial:int = 1, 
                 batch_size:int = 32,          
                 epoches:int = 100,
                 N:int = 5)->None:
    finetuner = FineTuner(dataset,base_path,model_path,output_path,log_path,trial,batch_size,epoches,N)
    finetuner.finetuning()

def main():
    # Load the fine-tuning datasets and other configurations
    DATASET = CFG['dataset'] # 'Drift'
    base_path = CFG['base_path']
    models_path = [str(os.path.join(base_path, model))for model in  CFG['models']]
    output_path = CFG['output_path']
    trials = CFG['trials']
    batch_size = CFG['batch_size']
    log_path = CFG['log_path']
    epoches = CFG['epoches']
    N = CFG['N']  

    if len(models_path) < 1 or len(trials) < 1:
        raise Exception('More than one model and trial must be given.')
    assert len(models_path) == len(trials)
    print(f'Got {len(models_path)} models to finetune.')
    
    for (model_path, trial) in tqdm(zip(models_path, trials), total=len(models_path)):
        run_finetuner(DATASET, base_path, 
                      model_path, ##
                      output_path, log_path, 
                      trial, ##
                      batch_size, epoches, N)

if __name__ == "__main__":
    main()