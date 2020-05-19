# TFM: APLICATIONS OF MACHINE LEARNING TO STUDIES OF QUANTUM PHASE TRANSITION

by Laura Malo Roset (April-September 2019).

Supervised by Alexandre Dauphin and co-supervised by Patrick Huembeli. 


# Objective
This repository contains all the codes needed in my Master Thesis. The project consists on creating datasets formed by wavefunction and density images 
to be trained and classified between the trivial and the topological phase according to the SSH model. Then, train different models with these datasets to be able to compare the results. And, finaly study the interpertability trhough heatmaps of the decision taken by the neural network. 

## Databases, training and interpreting

The project is maintly structured in three parts. 

  - Creating the databases: where we create the train, test and validation dataset for images labeled into to classes: trivial or topological. We create different variations of the dataset, considering disorder, permutations to be able to later perform methods such as transfer learning and data aumentation that are discussed in the report.  
  
 ![Non-disorder dataset example](https://github.com/laumalo/TFM/blob/master/images/database.png)
  
  - Training the model: we consider two models; a CNN trained from scratch and a ResNet model pretrained and used with the test data using transfer learning. 
  
 ![Image description](link-to-image)
  
 - Interpretability: once the prediction of the class has been made we used the CAM algorithm to interpret the decision and find the most discriminative regions of the image. 
 
![Image description](link-to-image)


## Results

The main results and further discussions of the project can be seen in the Master Thesis report, and more additional information can be found in the slides used for the public defense of the Master Thesis. 

[Report](https://github.com/laumalo/TFM/blob/master/TFM_LauraMalo.pdf) [Slides](https://github.com/laumalo/TFM/blob/master/TFM_LauraMalo.pdf)


## Codes

The codes included in the repository are the following: 

[``1-SSH_dataset.py`](https://github.com/laumalo/TFM/blob/master/My-CNN/1-SSH_dataset.py): this code uses the function hamiltonian() already defined in the previous notebook and computes the wavefunctions and the density in matrix (a function for
plotting this results for to examples is also impremented). Different methods for creating the datasets are used. The datasets are created following a structure of 
train/test/validation of 50/30/20. A record of all the datasets created is included (but commented) in the code. 

[`2-read_dataset.py`](https://github.com/laumalo/TFM/blob/master/My-CNN/2-read_dataset.py): reads the dataset created by SSH_dataset.py and loads it into a DataLoader to be able to train a CNN with these data. 
The aim of this code is only to test if the dataset is created in a correct way and the labels correspond to the images. 

[`3-cnn_models.py`](https://github.com/laumalo/TFM/blob/master/My-CNN/3-cnn_models.py): contains the definition of the different classes corresponding to each model used in the training, this script has to be imported in the training,testing and prediction script. 

[`4-training.py`](https://github.com/laumalo/TFM/blob/master/My-CNN/4-training.py): this code loads the datasets and trains a convolutional neural network to classify the images. The model used in each case (can be modified by
changing one line of the code) is imported from `ccn_model.py` where all the classes corresponding with the different scructures for the convolutional neural network 
are defined. The learned parameters of the learning are saved in a .pt file that will be used later. A summary of the training (its performance) is also created
as a .txt file.

[`5-test.py`](https://github.com/laumalo/TFM/blob/master/My-CNN/5-test.py): tests a model that has already been trained with a test set. The aim of this code is to be able to test images that are differents to the ones used 
during the training of the model. It returns only the summary of the test.

[`6-prediction+CAM.py`](https://github.com/laumalo/TFM/blob/master/My-CNN/6-prediction%2BCAM.py): once the model is trained using a specific dataset and a CNN structure, a model_.pt file is created with the model parameters. In this code
this parameters are read and used to make a prediction of images of the test set. Also the CAM technique is used in order to visualize the heatmap of the important
regions during the classification. 

The [ResNet18]() folder contains modified version of the codes `SSH_dataset.py`,`training.py`, `test.py` and `prediction+CAM.py` adapted to the pretrained NN **Resnet18()**. 
