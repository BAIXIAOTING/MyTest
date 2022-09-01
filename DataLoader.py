# load mitbih dataset

import os 
import numpy as np
import pandas as pd
import sys 
from tqdm import tqdm 

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import resample

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

cls_dit = {'normal':0, 'gas lock':1, 'gas affect':2,
                             'insufficient liquid supply':3, 'parting rod':4, 'valve leakage':5}

class mitbih_train(Dataset):
    def __init__(self, filename='/content/MyTest/power_train.csv', n_samples=2000, oneD=False):
        data_train = pd.read_csv(filename, header=None)
        
        # making the class labels for our dataset
        data_0 = data_train[data_train[187] == 0]
        data_1 = data_train[data_train[187] == 1]
        data_2 = data_train[data_train[187] == 2]
        data_3 = data_train[data_train[187] == 3]
        data_4 = data_train[data_train[187] == 4]
        data_5 = data_train[data_train[187] == 5]
        
        data_0_resample = resample(data_0, n_samples=n_samples, 
                                   random_state=123, replace=True)
        data_1_resample = resample(data_1, n_samples=n_samples, 
                                   random_state=123, replace=True)
        data_2_resample = resample(data_2, n_samples=n_samples, 
                                   random_state=123, replace=True)
        data_3_resample = resample(data_3, n_samples=n_samples, 
                                   random_state=123, replace=True)
        data_4_resample = resample(data_4, n_samples=n_samples, 
                                   random_state=123, replace=True)
        data_5_resample = resample(data_5, n_samples=n_samples, 
                                   random_state=123, replace=True)
        
        
        train_dataset = pd.concat((data_0_resample, data_1_resample, 
                         data_2_resample, data_3_resample, data_4_resample, data_5_resample))
        
        self.X_train = train_dataset.iloc[:, :-1].values
        if oneD:
            self.X_train = self.X_train.reshape(self.X_train.shape[0], 1, self.X_train.shape[1])
        else:
            self.X_train = self.X_train.reshape(self.X_train.shape[0], 1, 1, self.X_train.shape[1])
        self.y_train = train_dataset[187].values
            
        print(f'X_train shape is {self.X_train.shape}')
        print(f'y_train shape is {self.y_train.shape}')
        print(f'The dataset including {len(data_0_resample)} class 0, {len(data_1_resample)} class 1, {len(data_2_resample)} class 2, {len(data_3_resample)} class 3, {len(data_4_resample)} class 4, {len(data_5_resample)} class 5')
        
        
    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]
    
    
class mitbih_test(Dataset):
    def __init__(self, filename='/content/MyTest/power_test.csv', n_samples=600, oneD=False):
        data_test = pd.read_csv(filename, header=None)
        
        # making the class labels for our dataset
        data_0 = data_test[data_test[187] == 0]
        data_1 = data_test[data_test[187] == 1]
        data_2 = data_test[data_test[187] == 2]
        data_3 = data_test[data_test[187] == 3]
        data_4 = data_test[data_test[187] == 4]
        data_5 = data_test[data_test[187] == 5]
        
        data_0_resample = resample(data_0, n_samples=n_samples, 
                           random_state=123, replace=True)
        data_1_resample = resample(data_1, n_samples=n_samples, 
                                   random_state=123, replace=True)
        data_2_resample = resample(data_2, n_samples=n_samples, 
                                   random_state=123, replace=True)
        data_3_resample = resample(data_3, n_samples=n_samples, 
                                   random_state=123, replace=True)
        data_4_resample = resample(data_4, n_samples=n_samples, 
                                   random_state=123, replace=True)
        data_5_resample = resample(data_5, n_samples=n_samples, 
                                   random_state=123, replace=True)
        
        test_dataset = pd.concat((data_0_resample, data_1_resample, 
                                  data_2_resample, data_3_resample, data_4_resample))
        
        self.X_test = test_dataset.iloc[:, :-1].values
        if oneD:
            self.X_test = self.X_test.reshape(self.X_test.shape[0], 1, self.X_test.shape[1])
        else:
            self.X_test = self.X_test.reshape(self.X_test.shape[0], 1, 1, self.X_test.shape[1])
        self.y_test = test_dataset[187].values
        
        print(f'X_test shape is {self.X_test.shape}')
        print(f'y_test shape is {self.y_test.shape}')
        print(f'The dataset including {len(data_0)} class 0, {len(data_1)} class 1, {len(data_2)} class 2, {len(data_3)} class 3, {len(data_4)} class 4, {len(data_5)} class 5')
        
    def __len__(self):
        return len(self.y_test)
    
    def __getitem__(self, idx):
        return self.X_test[idx], self.y_test[idx]
