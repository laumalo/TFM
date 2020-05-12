"""
Created on Wed Jul  3 09:40:35 2019

@author: Laura Malo
"""

import __future__
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
from torch import nn, optim 
import torchvision
import torchvision.transforms as transforms
import time
import os
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils import data

#%%
# ------------------------------------------------------------------------------------------
#                               DEFINITION OF FUNCTIONS AND CLASSES 
# ------------------------------------------------------------------------------------------
    
def plot_examples(loader,fname):
    """Given a loader iterates into a batch and plots 10 examples images of the dataset selected. The
    plot of the 10 examples is saved in a file named fname."""
    dataiter = iter(loader)
    plt.figure()
    for i in range(10):
        images,labels = dataiter.next()
        images = images.unsqueeze(dim=1)
        img = images[i]
        plt.subplot(2,5,i+1)
        plt.title(classes[labels[i]])
        plt.imshow(img[0])
        plt.colorbar()
    plt.show()
    plt.savefig(os.path.join(os.getcwd(),'results',fname + '.png'))

# ------------------------------------------------------------------------------------------
      
class Dataset_Train(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels,transform = None):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = int(self.list_IDs[index])

        # Load data and get label
        X = torch.load(os.path.join(path_train,str(ID) + '.pt'))
        y = self.labels[ID]
        
        X = Variable(X).float()
        y = Variable(y).type(torch.LongTensor)
        return X, y

# ------------------------------------------------------------------------------------------
      
class Dataset_Valid(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels, transform = None):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = int(self.list_IDs[index])

        # Load data and get label
        X = torch.load(os.path.join(path_valid,str(ID) + '.pt'))
        y = self.labels[ID]
        
        X = Variable(X).float()
        y = Variable(y).type(torch.LongTensor)
        return X, y

# ------------------------------------------------------------------------------------------
      
class Dataset_Test(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels, transform = None):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = int(self.list_IDs[index])

        # Load data and get label
        X = torch.load(os.path.join(path_test,str(ID) + '.pt'))
        y = self.labels[ID]
        
        X = Variable(X).float()
        y = Variable(y).type(torch.LongTensor)
        return X, y   
    
#%%
        
    
        
#%%

# ------------------------------------------------------------------------------------------
#                                DEVICE AND HYPERPARAMETERS
# ------------------------------------------------------------------------------------------

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ------------------------------------------------------------------------------------------

# Hyper parameters
num_epochs = 10
batch_size = 100
learning_rate = 0.001
momentum= 0.8
weight_decay = 0.01 #add in the optim in order to implement L2 regularization

#Number of classes
num_classes = 2
classes = ['trivial', 'topological']

# ------------------------------------------------------------------------------------------

#select a dataset
dataset = input("Which dataset do you want to read:")

# ------------------------------------------------------------------------------------------

#paths for the diferent datasets
path =  os.path.join(os.getcwd(),'datasets',dataset) 
path_valid = os.path.join(path,'validation')
path_test = os.path.join(path,'test')
path_train =  os.path.join(path,'train')

#%%
# ------------------------------------------------------------------------------------------
#                                      LOAD THE DATASET
# ------------------------------------------------------------------------------------------

transform_dataset = transforms.Compose([
        transforms.Normalize((0.0,), (1.0,)) 
        ])   
    
#Generators training
labels_train = torch.load(os.path.join(path_train,'labels.pt'))
training_set = Dataset_Train(np.linspace(1,len(labels_train)-1, num = len(labels_train)), labels_train, transform_dataset)
train_loader = data.DataLoader(dataset = training_set, batch_size = batch_size, shuffle = True)

#Generators validation
labels_valid = torch.load(os.path.join(path_valid,'labels.pt'))
validation_set = Dataset_Valid(np.linspace(1,len(labels_valid)-1, num = len(labels_valid)), labels_valid, transform_dataset)
valid_loader = data.DataLoader(dataset = validation_set, batch_size = batch_size, shuffle = True)

#Generators testing
labels_test = torch.load(os.path.join(path_test,'labels.pt'))
testing_set = Dataset_Test(np.linspace(1,len(labels_test)-1, num = len(labels_test)), labels_test, transform_dataset)
test_loader = data.DataLoader(dataset = testing_set, batch_size = batch_size, shuffle = True)

#print the summary of the datasets
print('-----------------------------------------------------------')
print('                     SUMMARY                               ')
print('-----------------------------------------------------------')
print('DATASET:', dataset)
print('The dataset has been loaded!')
print('Training set/ Validation set/ Testing set')
print('Total number of data:', len(labels_train),'/',len(labels_valid),'/',len(labels_test))
print('Samples for training:',len(train_loader),'/', len(valid_loader),'/', len(test_loader))
print('-----------------------------------------------------------')

#plot a few examples of each dataset
plot_examples(loader = train_loader,fname = 'ex_trainset_' + dataset)
plot_examples(loader = valid_loader,fname = 'ex_validset_' + dataset)
plot_examples(loader = test_loader,fname = 'ex_testset_' + dataset)


