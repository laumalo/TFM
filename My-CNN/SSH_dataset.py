"""
Created on Wed Jun 26 15:49:47 2019

@author: Laura Malo 
"""

#import libraries
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import torch 
import os 

# ---------------------------------------------------------------------------------------------------
#                               DEFINITION OF FUNCTIONS
# ---------------------------------------------------------------------------------------------------

def hamiltonian_general(boundaries,M,v,w):
    """ Computes the hamiltonian of the SSH model for the following parameters:
    v -> intracell hopping amplitude
    w -> intercell hopping amplitude
    M -> number of cells
    boundaries -> 'periodic' (Born-von Karman) boundary conditions or 'open' boundaries"""
    diag = v*np.ones(2*M-1)
    diag[1::2] = w
    hamiltonian=np.diagflat(diag, 1) + np.diagflat(diag,-1)
    if boundaries=='periodic':
        hamiltonian[0,2*M-1]=w
        hamiltonian[2*M-1,0]=w
    return hamiltonian

# ---------------------------------------------------------------------------------------------------
    
def add_disorder(diag,W):
    """This function adds disorder to a diagonal of the hamiltonian given magnitude for the disorder 
    strength. """
    W1=W/2
    W2=W
    for i in range(0,len(diag)+1,2):
        R1=np.random.uniform(-0.5,0.5)
        R2=np.random.uniform(-0.5,0.5)
        diag[i] += W1*R1
        if (i+1)<(len(diag)): diag[i+1] += W2*R2
    return diag

# ---------------------------------------------------------------------------------------------------
    
def hamiltonian_disorder(boundaries,M,v,w,W):
    """ Computes the hamiltonian of the SSH model with disorder for the following parameters:
    v -> intracell hopping amplitude
    w -> intercell hopping amplitude
    M -> number of cells
    boundaries -> 'periodic' (Born-von Karman) boundary conditions or 'open' boundaries
    W -> disorder strength """
    diag = v*np.ones(2*M-1)
    diag[1::2] = w 
    diag_disorder = add_disorder(diag,W)
    hamiltonian=np.diagflat(diag_disorder, 1) + np.diagflat(diag_disorder,-1)
    if boundaries=='periodic':
        hamiltonian[0,2*M-1]=w
        hamiltonian[2*M-1,0]=w
    return hamiltonian
# ---------------------------------------------------------------------------------------------------
    
def magnitudes(boundaries,M,v,w):
    """Given the parameters of the system (boundaries, number of unit cells and the hopping
    amplitudes) returns a matrix with the wavefunctions and a matrix with the densities"""
    H1=hamiltonian_general(boundaries,M,v,w)
    eigdat1=LA.eigh(H1)    
    vecdat1=eigdat1[1]
    den1=np.abs(vecdat1)**2
    return vecdat1, den1 

# ---------------------------------------------------------------------------------------------------
def magnitudes_disorder(boundaries,M,v,w,W):
    """Given the parameters of the system (boundaries, number of unit cells and the hopping
    amplitudes and disorder) returns a matrix with the wavefunctions and a matrix with the densities"""
    H1=hamiltonian_disorder(boundaries,M,v,w,W) 
    eigdat1=LA.eigh(H1)    
    vecdat1=eigdat1[1]
    den1=np.abs(vecdat1)**2
    return vecdat1, den1 
# ---------------------------------------------------------------------------------------------------
    
def plots(boundaries,M,v,w,name):
    """Given the parameters of the system boundaries, number of unit cells and the hopping
    amplitudes) plots the Wavefunction and the Density profile."""
    vecdat1 = magnitudes(boundaries,M,v,w)[0]
    den1 = magnitudes(boundaries,M,v,w)[1]
    textstr = '\n'.join((
    r'$M=%.0f$' % (M, ),
    r'$\omega=%.2f$' % (w, ),
    r'$\nu=%.2f$' % (v, )))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.figure()
    plt.subplot(1,2,1)
    plt.title('Wavefunction')
    plt.imshow(vecdat1, vmin =-0.6, vmax=0.6, cmap='afmhot')
    plt.text(0.05, 0.95, textstr, fontsize=10,
        verticalalignment='top', bbox=props)
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.title('Density profile')
    plt.imshow(den1,vmin = 0,vmax =0.1, cmap='afmhot')
    plt.colorbar()
    plt.text(0.05, 0.95, textstr, fontsize=10,
        verticalalignment='top', bbox=props)
    plt.savefig(os.path.join(os.getcwd(),'results',name + '.png'))

# ---------------------------------------------------------------------------------------------------
    
def inversion(wf):
    new = []
    for i in range(len(wf)):
        new.append(wf[i][::-1])
    return new

# ---------------------------------------------------------------------------------------------------
    
def permutation(mat):
    num_colums = len(mat[0])
    your_permutation = np.random.permutation(num_colums)
    perm_mat = np.zeros((len(your_permutation), len(your_permutation)))
    for idx, i in enumerate(your_permutation):
        perm_mat[idx,i] = 1
    return np.dot(mat, perm_mat)
    
# ---------------------------------------------------------------------------------------------------

def save_pt_datasets(boundaries,M,ratios,type_image,folder):
    """Given the parameters of the system (boundary conditions and number of unit cells) and 
    a list of ratios for the v/w being v and w the two hopping amplitudes, creates as a .pt files 
    the images corresponding to the wavefunction/density(given as a parameter) and a file with a list of all 
    the labels of the images. The length of the dataset is given by the number of ratios provided to the 
    function.It creates three datasets for training(50%), testing(30%) and validation(20%) in diferent folders 
    /train/, /test/, /validation/ respectevely in /dataset/. The sequence for the data distribution is 
    T t T V T t T V T t repeatedly."""
    labels_train = []
    labels_valid = []
    labels_test = []
    count_train = 0.0
    count_valid = 0.0
    count_test = 0.0
    count = -1
    test_bool = np.array([True,False,True,False,True])
    
    if type_image == 'wavefunction': im = 0 
    if type_image == 'density': im = 1
    
    for i in range(len(ratios)):
        w = 1
        ratio = ratios[i]
        v= w*ratio
        if i%2 ==0 :    
            count_train += 1
            path_train = os.path.join(os.getcwd(),'datasets',folder,'train')
            fname = "%d"%count_train + ".pt"
            if ratio > 1: 
                labels_train.append(0) # 0 -> trivial 
                torch.save(torch.tensor(magnitudes(boundaries = boundaries, M = M, v = v, w = w)[im]), os.path.join(path_train,fname))
            if ratio < 1:
                labels_train.append(1) # 1 -> topological 
                torch.save(torch.tensor(magnitudes(boundaries = boundaries, M = M, v = v, w = w)[im]), os.path.join(path_train,fname))
            torch.save(torch.tensor(labels_train), os.path.join(path_train,'labels.pt'))
        else:
            count += 1
            if (count==5):count = 0
            if test_bool[count]:
                count_test +=1
                path_test = os.path.join(os.getcwd(),'datasets',folder,'test')
                fname = "%d"%count_test + ".pt"
                if ratio > 1: 
                    labels_test.append(0) # 0 -> trivial 
                    torch.save(torch.tensor(magnitudes(boundaries = boundaries, M = M, v = v, w = w)[im]), os.path.join(path_test,fname))
                if ratio < 1:
                    labels_test.append(1) # 1 -> topological 
                    torch.save(torch.tensor(magnitudes(boundaries = boundaries, M = M, v = v, w = w)[im]), os.path.join(path_test,fname))
                torch.save(torch.tensor(labels_test), os.path.join(path_test,'labels.pt'))
            if not test_bool[count]: 
                count_valid += 1
                path_valid = os.path.join(os.getcwd(),'datasets',folder,'validation')
                fname = "%d"%count_valid + ".pt"
                if ratio > 1: 
                    labels_valid.append(0) # 0 -> trivial 
                    torch.save(torch.tensor(magnitudes(boundaries = boundaries, M = M, v = v, w = w)[im]), os.path.join(path_valid,fname))
                if ratio < 1:
                    labels_valid.append(1) # 1 -> topological 
                    torch.save(torch.tensor(magnitudes(boundaries = boundaries, M = M, v = v, w = w)[im]), os.path.join(path_valid,fname))
                torch.save(torch.tensor(labels_valid), os.path.join(path_valid,'labels.pt'))

# ---------------------------------------------------------------------------------------------------
                
def save_pt_datasets_shuffle(boundaries,M,ratios,type_image,folder):
    """Given the parameters of the system (boundary conditions and number of unit cells) and 
    a list of ratios for the v/w being v and w the two hopping amplitudes, creates as a .pt files 
    the images corresponding to the wavefunction/density(given as a parameter) and a file with a list of all 
    the labels of the images. The length of the dataset is given by the number of ratios provided to the 
    function. It creates three datasets for training(50%), testing(30%) and validation(20%) in diferent folders 
    /train/, /test/, /validation/ respectevely in /dataset/. The values of the ratio are created and the shuffled in order
    to have the three datasets with random images without any type of order."""
    
    labels_train = []
    labels_valid = []
    labels_test = []

    num_train = int(0.5*len(ratios))
    num_test = int(0.3*len(ratios))
    num_validation = int(0.2*len(ratios))
    
    if type_image == 'wavefunction': im = 0 
    if type_image == 'density': im = 1
    
    count_train = -1.0
    count_test = -1.0
    count_valid = -1.0
    
    for i in range(len(ratios)):
        w = 1
        ratio = ratios[i]
        v= w*ratio
        if i< num_train:
            count_train += 1
            path_train = os.path.join(os.getcwd(),'datasets',folder,'train')
            fname = "%d"%count_train + ".pt"
            if ratio > 1: 
                labels_train.append(0) # 0 -> trivial 
                torch.save(torch.tensor(magnitudes(boundaries = boundaries, M = M, v = v, w = w)[im]), os.path.join(path_train,fname))
            if ratio < 1:
                labels_train.append(1)# 1 -> topological 
                torch.save(torch.tensor(magnitudes(boundaries = boundaries, M = M, v = v, w = w)[im]), os.path.join(path_train,fname))
        if i>=num_train and i<(num_train+num_test):
            count_test +=1
            path_test = os.path.join(os.getcwd(),'datasets',folder,'test')
            fname = "%d"%count_test + ".pt"
            if ratio > 1: 
                labels_test.append(0) # 0 -> trivial 
                torch.save(torch.tensor(magnitudes(boundaries = boundaries, M = M, v = v, w = w)[im]), os.path.join(path_test,fname))
            if ratio < 1:
                labels_test.append(1) # 1 -> topological 
                torch.save(torch.tensor(magnitudes(boundaries = boundaries, M = M, v = v, w = w)[im]), os.path.join(path_test,fname))       
        if i>=(num_test+num_train):
            count_valid += 1
            path_valid = os.path.join(os.getcwd(),'datasets',folder,'validation')
            fname = "%d"%count_valid + ".pt"
            if ratio > 1:
                labels_valid.append(0) # 0 -> trivial 
                torch.save(torch.tensor(magnitudes(boundaries = boundaries, M = M, v = v, w = w)[im]), os.path.join(path_valid,fname))
            if ratio < 1:
                labels_valid.append(1) # 1 -> topological 
                torch.save(torch.tensor(magnitudes(boundaries = boundaries, M = M, v = v, w = w)[im]), os.path.join(path_valid,fname))   
    torch.save(torch.tensor(labels_train), os.path.join(path_train,'labels.pt'))
    torch.save(torch.tensor(labels_test), os.path.join(path_test,'labels.pt'))  
    torch.save(torch.tensor(labels_valid), os.path.join(path_valid,'labels.pt'))

# ---------------------------------------------------------------------------------------------------
                
def save_pt_dataaugmentation(boundaries,M,ratios,type_image,folder):
    
    labels_train = []
    labels_valid = []
    labels_test = []

    num_train = int(0.5*len(ratios))
    num_test = int(0.3*len(ratios))
    num_validation = int(0.2*len(ratios))
    
    if type_image == 'wavefunction': im = 0 
    if type_image == 'density': im = 1
    
    count_train = -2.0
    count_test = -2.0
    count_valid = -2.0
    
    for i in range(len(ratios)):
        w = 1
        ratio = ratios[i]
        v= w*ratio
        if i< num_train:
            count_train += 2
            path_train = os.path.join(os.getcwd(),'datasets',folder,'train')
            fname = "%d"%count_train + ".pt"
            fname_inversion = "%d"%(count_train+1) + ".pt"
            if ratio > 1: 
                labels_train.append(0) # 0 -> trivial 
                torch.save(torch.tensor(magnitudes(boundaries = boundaries, M = M, v = v, w = w)[im]), os.path.join(path_train,fname))
                labels_train.append(0)
                torch.save(torch.tensor(inversion(magnitudes(boundaries = boundaries, M = M, v = v, w = w)[im])), os.path.join(path_train,fname_inversion))
            if ratio < 1:
                labels_train.append(1)# 1 -> topological 
                torch.save(torch.tensor(magnitudes(boundaries = boundaries, M = M, v = v, w = w)[im]), os.path.join(path_train,fname))
                labels_train.append(1)
                torch.save(torch.tensor(inversion(magnitudes(boundaries = boundaries, M = M, v = v, w = w)[im])), os.path.join(path_train,fname_inversion))
        if i>=num_train and i<(num_train+num_test):
            count_test +=2
            path_test = os.path.join(os.getcwd(),'datasets',folder,'test')
            fname = "%d"%count_test + ".pt"
            fname_inversion = "%d"%(count_test+1) + ".pt"
            if ratio > 1: 
                labels_test.append(0) # 0 -> trivial 
                torch.save(torch.tensor(magnitudes(boundaries = boundaries, M = M, v = v, w = w)[im]), os.path.join(path_test,fname))
                labels_test.append(0)
                torch.save(torch.tensor(inversion(magnitudes(boundaries = boundaries, M = M, v = v, w = w)[im])), os.path.join(path_test,fname_inversion))
            if ratio < 1:
                labels_test.append(1) # 1 -> topological 
                torch.save(torch.tensor(magnitudes(boundaries = boundaries, M = M, v = v, w = w)[im]), os.path.join(path_test,fname)) 
                labels_test.append(1)
                torch.save(torch.tensor(inversion(magnitudes(boundaries = boundaries, M = M, v = v, w = w)[im])), os.path.join(path_test,fname_inversion)) 
        if i>=(num_test+num_train):
            count_valid += 2
            fname_inversion = "%d"%(count_valid+1) + ".pt"
            path_valid = os.path.join(os.getcwd(),'datasets',folder,'validation')
            fname = "%d"%count_valid + ".pt"
            if ratio > 1:
                labels_valid.append(0) # 0 -> trivial 
                torch.save(torch.tensor(magnitudes(boundaries = boundaries, M = M, v = v, w = w)[im]), os.path.join(path_valid,fname))
                labels_valid.append(0) # 0 -> trivial 
                torch.save(torch.tensor(inversion(magnitudes(boundaries = boundaries, M = M, v = v, w = w)[im])), os.path.join(path_valid,fname_inversion))
            if ratio < 1:
                labels_valid.append(1) # 1 -> topological 
                torch.save(torch.tensor(magnitudes(boundaries = boundaries, M = M, v = v, w = w)[im]), os.path.join(path_valid,fname)) 
                labels_valid.append(1)
                torch.save(torch.tensor(inversion(magnitudes(boundaries = boundaries, M = M, v = v, w = w)[im])), os.path.join(path_valid,fname_inversion)) 
    torch.save(torch.tensor(labels_train), os.path.join(path_train,'labels.pt'))
    torch.save(torch.tensor(labels_test), os.path.join(path_test,'labels.pt'))  
    torch.save(torch.tensor(labels_valid), os.path.join(path_valid,'labels.pt'))
    
# ---------------------------------------------------------------------------------------------------
    
def save_pt_permutation(boundaries,M,ratios,type_image,folder):
    """Given the parameters of the system (boundary conditions and number of unit cells) and 
    a list of ratios for the v/w being v and w the two hopping amplitudes, creates as a .pt files 
    the images corresponding to the wavefunction/density(given as a parameter) and a file with a list of all 
    the labels of the images. The length of the dataset is given by the number of ratios provided to the 
    function. It creates three datasets for training(50%), testing(30%) and validation(20%) in diferent folders 
    /train/, /test/, /validation/ respectevely in /dataset/. The values of the ratio are created and the shuffled in order
    to have the three datasets with random images without any type of order."""
    
    labels_train = []
    labels_valid = []
    labels_test = []

    num_train = int(0.5*len(ratios))
    num_test = int(0.3*len(ratios))
    num_validation = int(0.2*len(ratios))
    
    if type_image == 'wavefunction': im = 0 
    if type_image == 'density': im = 1
    
    count_train = -1.0
    count_test = -1.0
    count_valid = -1.0
    
    for i in range(len(ratios)):
        w = 1
        ratio = ratios[i]
        v= w*ratio
        if i< num_train:
            count_train += 1
            path_train = os.path.join(os.getcwd(),'datasets',folder,'train')
            fname = "%d"%count_train + ".pt"
            if ratio > 1: 
                labels_train.append(0) # 0 -> trivial 
                torch.save(torch.tensor(permutation(magnitudes(boundaries = boundaries, M = M, v = v, w = w)[im])), os.path.join(path_train,fname))
            if ratio < 1:
                labels_train.append(1)# 1 -> topological 
                torch.save(torch.tensor(permutation(magnitudes(boundaries = boundaries, M = M, v = v, w = w)[im])), os.path.join(path_train,fname))
        if i>=num_train and i<(num_train+num_test):
            count_test +=1
            path_test = os.path.join(os.getcwd(),'datasets',folder,'test')
            fname = "%d"%count_test + ".pt"
            if ratio > 1: 
                labels_test.append(0) # 0 -> trivial 
                torch.save(torch.tensor(permutation(magnitudes(boundaries = boundaries, M = M, v = v, w = w)[im])), os.path.join(path_test,fname))
            if ratio < 1:
                labels_test.append(1) # 1 -> topological 
                torch.save(torch.tensor(permutation(magnitudes(boundaries = boundaries, M = M, v = v, w = w)[im])), os.path.join(path_test,fname))       
        if i>=(num_test+num_train):
            count_valid += 1
            path_valid = os.path.join(os.getcwd(),'datasets',folder,'validation')
            fname = "%d"%count_valid + ".pt"
            if ratio > 1:
                labels_valid.append(0) # 0 -> trivial 
                torch.save(torch.tensor(permutation(magnitudes(boundaries = boundaries, M = M, v = v, w = w)[im])), os.path.join(path_valid,fname))
            if ratio < 1:
                labels_valid.append(1) # 1 -> topological 
                torch.save(torch.tensor(permutation(magnitudes(boundaries = boundaries, M = M, v = v, w = w)[im])), os.path.join(path_valid,fname))   
    torch.save(torch.tensor(labels_train), os.path.join(path_train,'labels.pt'))
    torch.save(torch.tensor(labels_test), os.path.join(path_test,'labels.pt'))  
    torch.save(torch.tensor(labels_valid), os.path.join(path_valid,'labels.pt'))

# ---------------------------------------------------------------------------------------------------
    
def save_pt_permutation_state(boundaries,M,ratio,num_images,type_image,folder):
    """Given the parameters of the system (boundary conditions and number of unit cells) and 
    a list of ratios for the v/w being v and w the two hopping amplitudes, creates as a .pt files 
    the images corresponding to the wavefunction/density(given as a parameter) and a file with a list of all 
    the labels of the images. The length of the dataset is given by the number of ratios provided to the 
    function. It creates three datasets for training(50%), testing(30%) and validation(20%) in diferent folders 
    /train/, /test/, /validation/ respectevely in /dataset/. The values of the ratio are created and the shuffled in order
    to have the three datasets with random images without any type of order."""
    
    labels_train = []
    labels_valid = []
    labels_test = []

    num_train = int(0.5*num_images)
    num_test = int(0.3*num_images)
    num_validation = int(0.2*num_images)
    
    if type_image == 'wavefunction': im = 0 
    if type_image == 'density': im = 1
    
    count_train = -1.0
    count_test = -1.0
    count_valid = -1.0
    
    for i in range(num_images):
        w = 1
        v= w*ratio
        if i< num_train:
            count_train += 1
            path_train = os.path.join(os.getcwd(),'datasets',folder,'train')
            fname = "%d"%count_train + ".pt"
            if ratio > 1: 
                labels_train.append(0) # 0 -> trivial 
                torch.save(torch.tensor(permutation(magnitudes(boundaries = boundaries, M = M, v = v, w = w)[im])), os.path.join(path_train,fname))
            if ratio < 1:
                labels_train.append(1)# 1 -> topological 
                torch.save(torch.tensor(permutation(magnitudes(boundaries = boundaries, M = M, v = v, w = w)[im])), os.path.join(path_train,fname))
        if i>=num_train and i<(num_train+num_test):
            count_test +=1
            path_test = os.path.join(os.getcwd(),'datasets',folder,'test')
            fname = "%d"%count_test + ".pt"
            if ratio > 1: 
                labels_test.append(0) # 0 -> trivial 
                torch.save(torch.tensor(permutation(magnitudes(boundaries = boundaries, M = M, v = v, w = w)[im])), os.path.join(path_test,fname))
            if ratio < 1:
                labels_test.append(1) # 1 -> topological 
                torch.save(torch.tensor(permutation(magnitudes(boundaries = boundaries, M = M, v = v, w = w)[im])), os.path.join(path_test,fname))       
        if i>=(num_test+num_train):
            count_valid += 1
            path_valid = os.path.join(os.getcwd(),'datasets',folder,'validation')
            fname = "%d"%count_valid + ".pt"
            if ratio > 1:
                labels_valid.append(0) # 0 -> trivial 
                torch.save(torch.tensor(permutation(magnitudes(boundaries = boundaries, M = M, v = v, w = w)[im])), os.path.join(path_valid,fname))
            if ratio < 1:
                labels_valid.append(1) # 1 -> topological 
                torch.save(torch.tensor(permutation(magnitudes(boundaries = boundaries, M = M, v = v, w = w)[im])), os.path.join(path_valid,fname))   
    torch.save(torch.tensor(labels_train), os.path.join(path_train,'labels.pt'))
    torch.save(torch.tensor(labels_test), os.path.join(path_test,'labels.pt'))  
    torch.save(torch.tensor(labels_valid), os.path.join(path_valid,'labels.pt'))
# ---------------------------------------------------------------------------------------------------
                
def save_pt_disorder(boundaries,M,ratios,type_image,folder):
    """Given the parameters of the system (boundary conditions and number of unit cells) and 
    a list of ratios for the v/w being v and w the two hopping amplitudes, creates as a .pt files 
    the images corresponding to the wavefunction/density(given as a parameter) and a file with a list of all 
    the labels of the images. The length of the dataset is given by the number of ratios provided to the 
    function. It creates three datasets for training(50%), testing(30%) and validation(20%) in diferent folders 
    /train/, /test/, /validation/ respectevely in /dataset/. The values of the ratio are created and the shuffled in order
    to have the three datasets with random images without any type of order."""
    
    labels_train = []
    labels_valid = []
    labels_test = []

    num_train = int(0.5*len(ratios))
    num_test = int(0.3*len(ratios))
    num_validation = int(0.2*len(ratios))
    
    if type_image == 'wavefunction': im = 0 
    if type_image == 'density': im = 1
    
    count_train = -1.0
    count_test = -1.0
    count_valid = -1.0
    
    W = input('Disorder strengh:')
    for i in range(len(ratios)):
        w = 1
        ratio = ratios[i]
        v= w*ratio
        if i< num_train:
            count_train += 1
            path_train = os.path.join(os.getcwd(),'datasets',folder,'train')
            fname = "%d"%count_train + ".pt"
            if ratio > 1: 
                labels_train.append(0) # 0 -> trivial 
                torch.save(torch.tensor(magnitudes_disorder(boundaries = boundaries, M = M, v = v, w = w, W = W)[im]), os.path.join(path_train,fname))
            if ratio < 1:
                labels_train.append(1)# 1 -> topological 
                torch.save(torch.tensor(magnitudes_disorder(boundaries = boundaries, M = M, v = v, w = w, W = W)[im]), os.path.join(path_train,fname))
        if i>=num_train and i<(num_train+num_test):
            count_test +=1
            path_test = os.path.join(os.getcwd(),'datasets',folder,'test')
            fname = "%d"%count_test + ".pt"
            if ratio > 1: 
                labels_test.append(0) # 0 -> trivial 
                torch.save(torch.tensor(magnitudes_disorder(boundaries = boundaries, M = M, v = v, w = w, W = W)[im]), os.path.join(path_test,fname))
            if ratio < 1:
                labels_test.append(1) # 1 -> topological 
                torch.save(torch.tensor(magnitudes_disorder(boundaries = boundaries, M = M, v = v, w = w, W = W)[im]), os.path.join(path_test,fname))       
        if i>=(num_test+num_train):
            count_valid += 1
            path_valid = os.path.join(os.getcwd(),'datasets',folder,'validation')
            fname = "%d"%count_valid + ".pt"
            if ratio > 1:
                labels_valid.append(0) # 0 -> trivial 
                torch.save(torch.tensor(magnitudes_disorder(boundaries = boundaries, M = M, v = v, w = w, W = W)[im]), os.path.join(path_valid,fname))
            if ratio < 1:
                labels_valid.append(1) # 1 -> topological 
                torch.save(torch.tensor(magnitudes_disorder(boundaries = boundaries, M = M, v = v, w = w, W = W)[im]), os.path.join(path_valid,fname))   
    torch.save(torch.tensor(labels_train), os.path.join(path_train,'labels.pt'))
    torch.save(torch.tensor(labels_test), os.path.join(path_test,'labels.pt'))  
    torch.save(torch.tensor(labels_valid), os.path.join(path_valid,'labels.pt')) 
    

    
            
#%%        
# ---------------------------------------------------------------------------------------------------
#                                                MAIN CODE    
# ---------------------------------------------------------------------------------------------------

#total number of desired images of the dataset
total_images = 20000

#ordered ratios 
ratios = np.linspace(0,2,total_images) 
ratios_transition = np.concatenate((np.linspace(0,0.5,int(total_images/2)),np.linspace(1.5,2,int(total_images/2))))

#shuffled ratios 
ratios_shuffle = np.linspace(0,2,total_images) 
np.random.shuffle(ratios_shuffle)

#ratios in the region of the deep phases
ratios_deep = np.concatenate((np.linspace(0,0.5,int(total_images/2)),np.linspace(1.5,2,int(total_images/2))))
np.random.shuffle(ratios_deep)

#ratios in the region of the phase transition
ratios_transition = (np.linspace(0.5,1.5,int(total_images)))
np.random.shuffle(ratios_transition)


#%% 
# ---------------------------------------------------------------------------------------------------
#                                               CREATES A SELECTED DATASET/S  
# ---------------------------------------------------------------------------------------------------


save_pt_permutation(boundaries = 'open' ,M = 30 ,ratios = ratios_deep, type_image = 'density' ,folder = 'den_60_perm_deep')



