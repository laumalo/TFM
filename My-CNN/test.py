"""
Created on Thu Jul  4 11:26:51 2019

@author: laura
"""

import __future__
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
from torch import optim 
import torchvision
import torchvision.transforms as transforms
import time
import os
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils import data
import cnn_models as models


#%%
# ------------------------------------------------------------------------------------------
#                               DEFINITION OF FUNCTIONS AND CLASSES 
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
      
class Dataset_Test(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

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
#                                   DATASET AND MODEL
# ------------------------------------------------------------------------------------------

#ask the user
dataset = input("Dataset with the test images you want to use:")
model_name = input("Trained model:")


#%%
# ------------------------------------------------------------------------------------------
#                                  LOAD THE DATASET (ONLY TEST FOLDER)
# ------------------------------------------------------------------------------------------
transform_dataset = transforms.Compose([
        transforms.Normalize((0.0,), (1.0,)) 
        ])    
    
#paths for the diferent datasets
path =  os.path.join(os.getcwd(),'datasets',dataset) 
path_test = os.path.join(path,'test')

#Generators testing
labels_test = torch.load(os.path.join(path_test,'labels.pt'))
testing_set = Dataset_Test(np.linspace(1,len(labels_test)-1, num = len(labels_test)), labels_test)
test_loader = data.DataLoader(dataset = testing_set, batch_size = batch_size, shuffle = True)


#%%

# ------------------------------------------------------------------------------------------
#                              LOAD THE PARAMETERS OF THE TRAINED MODEL
# ------------------------------------------------------------------------------------------

#class of the trained model (imported from cnn_models.py)
model = models.CNN2()

#path of the saved model parameters
PATH = os.path.join(os.getcwd(),'models',model_name + '.pt')
model.load_state_dict(torch.load(PATH, map_location='cpu'))

model = model.to(device)
#model in mode evaluation
model.eval()

#%%

# ------------------------------------------------------------------------------------------
#                                   TEST THE MODEL ON THE TEST SET
# ------------------------------------------------------------------------------------------

print('-----------------------------------------------------------')
print('Testing the model...')

time_ini_test = time.time()
with torch.no_grad():
    correct = 0
    total = 0
for i in range(num_epochs):
  for images, labels in test_loader:
        images = images.unsqueeze(dim=1).to(device)
        labels = labels.to(device)
        outputs = model(images)
        #Gets predictions for the maximum value
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

time_test=time.time() - time_ini_test
print('Testing completed in {:.0f}m {:.0f}s'.format(
        time_test // 60, time_test % 60))
print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))

print('-----------------------------------------------------------')

# ------------------------------------------------------------------------------------------
