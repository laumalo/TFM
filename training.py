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
import cnn_models as models



#%%
# ------------------------------------------------------------------------------------------
#                               DEFINITION OF FUNCTIONS AND CLASSES 
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

# ------------------------------------------------------------------------------------------
        
def printing(text, f):
    """We define a new print functionin which when used this command the print is done in the console
    and also in a .txt file. This is used for the summary of the model (training and testing) to be able to 
    visualized it at the moment and to save the performanc of the model."""
    print(text)
    print(text, file = f)
    
#%%
# ------------------------------------------------------------------------------------------
#                                DEVICE AND HYPERPARAMETERS
# ------------------------------------------------------------------------------------------

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ------------------------------------------------------------------------------------------

# Hyper parameters
num_epochs = 50
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

#Dataset and model (ask the user)

dataset = input("Which dataset do you want to read:")
model_name = input("How do you want to save the model:")

# ------------------------------------------------------------------------------------------
#                                   PATHS AND FILES
# ------------------------------------------------------------------------------------------

#path to save the model
PATH_MODEL = os.path.join(os.getcwd(),'models')

#dataset paths
path =  os.path.join(os.getcwd(),'datasets',dataset) 
path_valid = os.path.join(path,'validation')
path_test = os.path.join(path,'test')
path_train =  os.path.join(path,'train')

#file where a summary of the training is saved
filename = os.path.join(PATH_MODEL,'summary_'+str(model_name)+'.txt')
filename_plots = os.path.join(PATH_MODEL,'plots_'+str(model_name)+'.txt')

f = open(filename,'w')
f_plots = open(filename_plots, 'w')

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
printing('-----------------------------------------------------------',f)
printing('                     SUMMARY                               ',f)
printing('-----------------------------------------------------------',f)
printing({'DATASET:', dataset},f)
printing('The dataset has been loaded!',f)
printing('Training set/ Validation set/ Testing set',f)
printing({'Total number of data:', len(labels_train),'/',len(labels_valid),'/',len(labels_test)},f)
printing({'Samples for training:',len(train_loader),'/', len(valid_loader),'/', len(test_loader)},f)

#%%
# ------------------------------------------------------------------------------------------
#                                       MODEL
# ------------------------------------------------------------------------------------------

# define the model and convert their parameters and buffers to CUDA tensors (if available)
model = models.CNN2(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss() #cross entropy loss
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, 
                            weight_decay = weight_decay) 

#%%
# ------------------------------------------------------------------------------------------
#                                   TRAINING THE MODEL
# ------------------------------------------------------------------------------------------

total_step = len(train_loader)
count = 0.0
count_list = []
loss_list = []
loss_val_list = []
accuracy_list = []
accuracy_val_list = []
time_ini=time.time()
print('-----------------------------------------------------------')
print('Training beggins...')
for epoch in range(num_epochs):
        print({'Epoch {:.0f}/ {:.0f}'.format(
            epoch+1, num_epochs)})
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
                #send them to device
                images = images.unsqueeze(dim=1).to(device) 
                labels = labels.to(device)
                # Forward pass (we pass all the elements in images through the model)
                outputs = model(images)
                loss = criterion(outputs, labels)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                #update parameters
                optimizer.step()  
                _,predicted = torch.max(outputs.data,1)
                total += len(labels)
                correct += (predicted == labels).sum()
                accuracy = 100 * float(correct) / float(total)
                # print statistics
                count += 1
                if count % 50 == 0:
                  count_list.append(count)
                  # Calculate Accuracy         
                  correct_val = 0
                  total_val = 0
                  # Iterate through test dataset
                  loss_val = 0.0
                  for images, labels in valid_loader:
                    images_valid = images.unsqueeze(dim=1).to(device)
                    labels_valid = labels.to(device)
                    # Forward propagation
                    outputs = model(images_valid)
                    loss_val = criterion(outputs,labels_valid)  
                    # Get predictions from the maximum value
                    _,predicted = torch.max(outputs.data, 1)
                    # Total number of labels
                    total_val += len(labels_valid)
                    correct_val += (predicted == labels_valid).sum()
                  accuracy_val = 100 * float(correct_val) / float(total_val)
                  
            
                  # store loss and accuracy
                  loss_list.append(loss.data)
                  loss_val_list.append(loss_val.data)
                  accuracy_list.append(accuracy)
                  accuracy_val_list.append(accuracy_val)


printing('Finished Training!',f)
printing({'Final accuracy:',accuracy},f)

#how much time has the training last
time_train=time.time() - time_ini
printing({'Training completed in {:.0f}m {:.0f}s'.format(
        time_train // 60, time_train % 60)},f)

#%%
# ------------------------------------------------------------------------------------------
#                                   PERFORMANCE OF THE TRAINING
# ------------------------------------------------------------------------------------------

print(count_list, file = f_plots)
print(accuracy_list, file = f_plots)
print(accuracy_val_list, file = f_plots)

print(loss_list, file = f_plots)
print(loss_val_list, file = f_plots)

#%%
# ------------------------------------------------------------------------------------------
#                                   SAVE THE TRAINED MODEL
# ------------------------------------------------------------------------------------------

torch.save(model.state_dict(), os.path.join(PATH_MODEL,model_name + '.pt'))

#%%
# ------------------------------------------------------------------------------------------
#                                   TEST THE MODEL ON THE SAME DATASET (TEST FOLDER)
# ------------------------------------------------------------------------------------------

printing('-----------------------------------------------------------',f)
printing('Testing the model...',f)
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
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
printing({'Testing completed in {:.0f}m {:.0f}s'.format(
        time_test // 60, time_test % 60)},f)
printing('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total),f)

printing('-----------------------------------------------------------',f)

f.close()