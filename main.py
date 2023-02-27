# -*- coding: utf-8 -*-
"""
@Project: AEDCoder20211214
@File:    UAED
@Author:  Jiaqi
@Time:    2023/02/23 15:02
@Description: The UAED model for Unsupervised Abnormal Emotion Detection,
using a stacking operation to increase the dimension of the one-dimensional ECG signal,
using CNN layers,
using VAE with Gaussian mixture distribution prior,
can weight the anomaly score or not
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from matplotlib import pyplot as plt
import numpy as np
import vae_cnn     
import gmm_prior
import os
from torch.utils.data import DataLoader,Dataset,TensorDataset
from torchvision.utils import save_image,make_grid
import torch
from sklearn import preprocessing
import time
from PIL import Image
from thop import profile
from torchstat import stat

warnings.filterwarnings("ignore")

class Datatox(Dataset):
    def __init__(self, datasets ,labels:list):
        self.dataset = [datasets[i][0] for i in range(len(datasets)) 
                        if datasets[i][1] in labels ]   
        self.labels = labels
        self.len_oneclass = int(len(self.dataset)/10)
        
    def __len__(self):
        return int(len(self.dataset))

    def __getitem__(self, index):
        img = self.dataset[index]
        scaler = preprocessing.MinMaxScaler()
        img = torch.tensor(scaler.fit_transform(img.squeeze(0))).float().unsqueeze(0)
        return img,[]


def train_model(mutual_iteration,  dataloader, all_loader, fixed_x, bs, k, model_dir ):
    ARI_list = np.zeros((mutual_iteration,1))
    for i in range(mutual_iteration):
        print(f"------------------mutual_iteration:{i+1}------------------")  
        if i == 0: 
            gmm_mu = None; gmm_var = None
            
        z = vae_cnn.train(iteration=i, gmm_mu=gmm_mu, gmm_var=gmm_var, 
                           epochs=200 , train_loader=dataloader, all_loader=all_loader,  
                           fixed_x=fixed_x, batchsize=bs, model_dir=model_dir )  
        gmm_mu, gmm_var, max_ARI = gmm_prior.train(iteration=i, x_d = z, K=k, epoch=100)     
        gmm_mu = torch.Tensor(gmm_mu)
        gmm_var = torch.Tensor(gmm_var)
        ARI_list[it][0]=max_ARI
    return z 

bs = 24                      #batch size
k = 8                        #clusters in the Gaussian distribution
z_dim = 32                   #dimensions of vae's latent space
lam = 0.1                    #trade-off parameter of the anomaly score
n_k = 10                     # k-fold
mutual_iteration = 5
model_dir = "./model/debug"
data_dir = "./datasets"
current_dir = 'D:\zjq\UAEDCoder20220420\Pytorch_VAE-GMM-main_stack'
os.chdir(current_dir)       #Change the path
# print(os.getcwd())

#WESAD  for each participant
if os.path.exists(data_dir):    #LOSO
    Xtrain_ECG = np.load(data_dir+"/us_Xtrain_ECG.npy")   
    ytrain_ECG = np.load(data_dir+"/us_ytrain_ECG.npy")   
    Xtest_ECG = np.load(data_dir+"/s1_Xtest_ECG.npy")      
    ytest_ECG = np.load(data_dir+"/s1_ytest_ECG.npy")
print("Xtrain's shape:", Xtrain_ECG.shape,"ytrain's shape:", ytrain_ECG.shape, "Xtest's shape:", Xtest_ECG.shape,"ytest's shape:", ytest_ECG.shape)

#10-fold cross-validation
mun_validation_samples = int((len(Xtrain_ECG)-sum(ytrain_ECG)) // n_k)
Xtrain_n = Xtrain_ECG[ytrain_ECG==0]
sum=0
for fold in range(n_k):  
    validation_data = np.concatenate([Xtrain_n[mun_validation_samples*fold:mun_validation_samples*(fold+1)],
                                     Xtrain_ECG[ytrain_ECG==1]])
    validation_data_label = np.hstack((np.zeros(Xtrain_n[mun_validation_samples*fold:mun_validation_samples*(fold+1)].shape[0]), 
                                        np.ones(Xtrain_ECG[ytrain_ECG==1].shape[0]) ))
    a = Xtrain_n[:mun_validation_samples * fold]
    b = Xtrain_n[mun_validation_samples * (fold+1):]
    training_data=np.append(a,b,axis=0)
    train_dataset = TensorDataset(torch.from_numpy(training_data).float(),torch.from_numpy(np.zeros(training_data.shape[0])).float())
    Xtrain = Datatox(train_dataset,[0]) 
    dataloader = DataLoader(Xtrain, batch_size=bs, shuffle=False)
    all_loader = DataLoader(Xtrain, batch_size=len(training_data), shuffle=False)
    # Fixed input for debugging
    fixed_x,_ = next(iter(dataloader)) 
    # save_image(fixed_x, 'results/real_image.png')
    z = train_model(mutual_iteration,  dataloader, all_loader, fixed_x, bs, k, dir_name)
    #Validating
    Validate_dataset = TensorDataset(torch.from_numpy(validation_data).float(),torch.from_numpy(validation_data_label).float())
    XValidate = Datatox(Validate_dataset,[1,0]) 
    Validate_loader = DataLoader(XValidate, batch_size=1, shuffle=False)    
    validation_score_accuracy = vae_cnn.test(mutual_iteration, Validate_loader ,validation_data_label, lam, model_dir=model_dir)
    sum = sum + validation_score_accuracy
validation_score_average = sum/ n_k      #roc
print('Validation_score_average:', validation_score_average)

test_dataset = TensorDataset(torch.from_numpy(Xtest_ECG).float(),torch.from_numpy(ytest_ECG).float())
Xtest = Datatox(test_dataset,[1,0]) 
test_loader = DataLoader(Xtest, batch_size=1, shuffle=False)
vae_cnn.test(mutual_iteration, test_loader , ytest_ECG, model_dir=model_dir)

# Visualize normal and abnormal samples
test_loader_print = DataLoader(Xtest, batch_size=16, shuffle=False)
fixed_xn,_ = next(iter(test_loader_print))
save_image(fixed_xn, 'results/normal_image_our.png', 8, 2)

test_dataset2 = TensorDataset(torch.from_numpy(Xtest_ECG2).float(),torch.from_numpy(ytest_ECG).float())
Xtest2 = Datatox(test_dataset2,[1,0]) 
test_loader_print2 = DataLoader(Xtest2, batch_size=16, shuffle=False)
fixed_xa,_ = next(iter(test_loader_print2))
save_image(fixed_xa, 'results/abnormal_image_our.png', 8, 2)



