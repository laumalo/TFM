
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
num_epochs = 15
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
"""We ask the user which dataset wants to test"""
dataset = input("Dataset used:")
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

#model in mode evaluation
model.eval()

#%%
# ------------------------------------------------------------------------------------------
#                                   PREDICTIONS 
# ------------------------------------------------------------------------------------------

#selects a random image of the batch
idx_img = np.random.randint(low=1, high=99, size=1)[0]


print('-----------------------------------------------------------')
dataiter = iter(test_loader)
#to select an image of a batch 

images,labels = dataiter.next()
print('Label of the selected image:', classes[int(labels[idx_img])])
images = images.unsqueeze(dim=1)
img = images[idx_img]

img_variable = Variable(img.unsqueeze(0))

finalconv_name='layer5' #last convolutional layer of our model

#take the activated features of the last convolutional layer
features_layer=[]
def hook_feature(module, input, output):
    features_layer.append(output.data.cpu().numpy())

#we attach a hook to the last layer
model._modules.get(finalconv_name).register_forward_hook(hook_feature)

#get the softmax weight of the fc layer (on the side of the avarage pooling)
params=list(model.parameters())
weight_softmax=torch.squeeze(params[-2])

#we pass the image through the hole CNN
logit = model(img_variable)

#we make the prediction of the image for each class
h_x = F.softmax(logit, dim=1).data.squeeze()
probs, idx = h_x.sort(0, True)
probs = probs.cpu().numpy()
idx = idx.cpu().numpy()

# output the prediction (only for the top 5 classes)
print('Top predictions for the image:')
for i in range(0, 2):
    print('{:.5f} -> {}'.format(probs[i], classes[idx[i]]))
    

#%%

#%%
# ------------------------------------------------------------------------------------------
#                                   CAM AND HEATMAP
# ------------------------------------------------------------------------------------------

def getCAM(feature_conv,wight_softmax,class_idx):
  """we index into the fully connected layer to get the weights for that class 
  and calculate the dot product with our features from the image. """
  bz,nc,h,w = (feature_conv[0]).shape #dimensions of the features
  for idx in class_idx: #class_idx we want to investigate the heatmap
    features=torch.Tensor((feature_conv[0]).reshape((nc,h*w)))
    #matricial product between the weigths and the features of the image
    cam = torch.mm(weight_softmax[idx].reshape((1,len(weight_softmax[idx]))),features)
    cam = cam.reshape(h,w)
    cam = cam - torch.min(cam)
    cam_img = cam/torch.max(cam)
  return [cam_img]


# generate class activation mapping for the top1 prediction
CAMs = getCAM(features_layer, weight_softmax, [idx[0]])
CAMs[0]

#CAMs is a list with only one element in the 0 position
#as we haven't detach the hook we have to use .detach()
#to do the plot, save in cpu and convert to numpy array
heatmap= CAMs[0].detach().cpu().numpy()

#heatmap
plt.figure()
plt.subplot(1,3,2)
plt.imshow(heatmap)
textstr = '\n'.join((
    r'{:.5f} -> {}'.format(probs[0], classes[idx[0]]),
    r'{:.5f} -> {}'.format(probs[1], classes[idx[1]])))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.00, 0.00, textstr, fontsize=10,
        verticalalignment='top', bbox=props)

#original image
plt.subplot(1,3,1)
plt.imshow(img[0], vmin = 0, vmax = 0.1, cmap='afmhot')
plt.title(classes[labels[idx_img]])

#heatmap on top of the original image
plt.subplot(1,3,3)
plt.imshow(img[0],cmap='afmhot', alpha=1) # for image
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
plt.imshow(heatmap, interpolation='nearest', alpha=.8, extent=(xmin,xmax,ymin,ymax))   # for heatmap to overlap
plt.savefig(os.path.join(os.getcwd(),'results', model_name + '_im' + str(idx_img) + '_heatmap.png'))

