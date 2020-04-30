# TFM: APLICATIONS OF MACHINE LEARNING TO STUDIES OF QUANTUM PHASE TRANSITION

by Laura Malo Roset (April-September 2019).

Supervised by Alexandre Dauphin and co-supervised by Patrick Huembeli. 


# Su-Schrieffer-Heeger (SSH) model
This repository contains all the codes needed in the SSH project(TFM). The SSH project consists on creating datasets formed by wavefunction and density images 
to be trained and classified between the trivial and the topological phase according to the SSH model.

Two approaches are taking into account to be compared; the used of a pretrained ResNet18 NN, a CNN trained by us with the structure created by adjusting the 
hyperparamenters and the layers. A more simpler training is done (the one on the paper) and a training with fast.ai library is also studied. The second approach is 
using the ResNet18 structure, here we use a pretrained ResNet18 and is compared with a ResNet18 we train. 

For the master thesis report the results and codes used are the corresponding with the more 'simple' training for the CNN and the pretrained ResNet18.

## My-CNN

The codes included in the repository are the following: 

`SSH_model.ipnyb` : this notebook contains the code to reproduce the main figures shown in the course on Topological insulators ¡. 

`SSH_dataset.py`: this code uses the function hamiltonian() already defined in the previous notebook and computes the wavefunctions and the density in matrix (a function for
plotting this results for to examples is also impremented). Different methods for creating the datasets are used. The datasets are created following a structure of 
train/test/validation of 50/30/20. A record of all the datasets created is included (but commented) in the code. 

`read_dataset.py`: reads the dataset created by SSH_dataset.py and loads it into a DataLoader to be able to train a CNN with these data. 
The aim of this code is only to test if the dataset is created in a correct way and the labels correspond to the images. 

`training.py`: this code loads the datasets and trains a convolutional neural network to classify the images. The model used in each case (can be modified by
changing one line of the code) is imported from `ccn_model.py` where all the classes corresponding with the different scructures for the convolutional neural network 
are defined. The learned parameters of the learning are saved in a .pt file that will be used later. A summary of the training (its performance) is also created
as a .txt file.

`test.py`: tests a model that has already been trained with a test set. The aim of this code is to be able to test images that are differents to the ones used 
during the training of the model. It returns only the summary of the test.

`prediction+CAM.py`: once the model is trained using a specific dataset and a CNN structure, a model_.pt file is created with the model parameters. In this code
this parameters are read and used to make a prediction of images of the test set. Also the CAM technique is used in order to visualize the heatmap of the important
regions during the classification. 


## ResNet18 

The ResNet18 folder contains modified version of the codes `SSH_dataset.py`,`training.py`, `test.py` and `prediction+CAM.py` adapted to the pretrained NN **Resnet18()**. This scripts are `training_resnet.py` and `prediction+CAM_resnet.py`.
