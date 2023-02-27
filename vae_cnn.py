# -*- coding: utf-8 -*-
"""
@Project: AEDCoder20211214
@File:    UAED
@Author:  Jiaqi
@Description: CNN-based VAE for anomaly emotion detection on WESAD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
# from torchsummary import summary
from torch.utils.data import DataLoader,Dataset,TensorDataset
from random import randint
import os
from sklearn import preprocessing
from IPython.display import Image
from IPython.core.display import Image, display
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,auc,precision_recall_curve,roc_curve
import warnings
import numpy as np
import matplotlib.pyplot as plt
from torchstat import stat
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture
from thop import profile
import time 
from torch.distributions import Normal, kl_divergence

warnings.filterwarnings('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Mnisttox(Dataset):
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

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class UnFlatten(nn.Module):
    def forward(self, input, size=64):
        return input.view(input.size(0), size, 2, 2)
    
    
class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=64*2*2, z_dim=32):   
        super(VAE, self).__init__()   
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 8, kernel_size=11, stride=2), 
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),  
            nn.Conv2d(8, 16, kernel_size=5, stride=2),  
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),  
            nn.Conv2d(16, 32, kernel_size=5, stride=2),  
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),  
            nn.Conv2d(32, 64, kernel_size=5, stride=2),  
            nn.BatchNorm2d(64),
        )
        self.ff = Flatten()  
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2,output_padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2,output_padding=1), 
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2), 
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, image_channels, kernel_size=11, stride=2 ,output_padding=1), 
            nn.Sigmoid(),
        )
        
        # N(0,I) initialize
        self.prior_var = nn.Parameter(torch.Tensor(1, z_dim).float().fill_(1.0))
        self.prior_logvar = nn.Parameter(self.prior_var.log())
        self.prior_var.requires_grad = False
        self.prior_logvar.requires_grad = False
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_().to(device) 
        esp = torch.randn(*mu.size()).to(device)  
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        h = self.ff(h)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        x_hat = self.decode(z)
        mu_hat = self.decode(mu)
        logvar_hat = self.decode(logvar)
        return x_hat, mu, logvar, z , mu_hat, logvar_hat

    def loss_fn(self, recon_x, x, en_mu, en_logvar, gmm_mu, gmm_var, iteration , mu_hat, logvar_hat,test=False):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        if iteration != 0:     
            gmm_mu = nn.Parameter(gmm_mu)
            prior_mu = gmm_mu
            prior_mu.requires_grad = False
            prior_mu = prior_mu.expand_as(en_mu).to(device).to(torch.float32)
            
            gmm_var = nn.Parameter(gmm_var)
            prior_var = gmm_var
            prior_var.requires_grad = False
            prior_var = prior_var.expand_as(en_logvar).to(device).to(torch.float32)
            
            prior_logvar = nn.Parameter(prior_var.log())
            prior_logvar.requires_grad = False
            prior_logvar = prior_logvar.expand_as(en_logvar).to(device).to(torch.float32)
            
            var_division = en_logvar.exp() / prior_var # Σ_0 / Σ_1    ####
            diff = en_mu - prior_mu # μ_１ - μ_0
            diff_term = diff *diff / prior_var  # (μ_1 - μ_0)(μ_1 - μ_0)/Σ_1    ####
            logvar_division = prior_logvar - en_logvar # log|Σ_1| - log|Σ_0| = log(|Σ_1|/|Σ_2|)    ####
            KLD = 0.5 * ( torch.mean((var_division + diff_term + logvar_division).sum(1) )- z_dim  )
        else:        
            KLD = -0.5 * torch.sum(1 + en_logvar - en_mu.pow(2) - en_logvar.exp())   
        return BCE + KLD, BCE, KLD  


def send_all_z(iteration, all_loader, model_dir="./vae_gmm"): 
    model = VAE().to(device)
    model.load_state_dict(torch.load(model_dir+"/pth/vae_"+str(iteration)+".pth"))
    model.eval()
    for idx, [images,_] in enumerate(all_loader):
        images = images.to(device)
        recon_images, mu, logvar, z,_ = model(images)
        z = z.cpu()
    return z.detach().numpy()


def train(iteration, gmm_mu, gmm_var, epochs, train_loader, all_loader, fixed_x, batchsize, model_dir="./vae_gmm"):    
    print("VAE Training Start")  
    image_channels = fixed_x.size(1)
    loss_list = np.zeros((epochs))
    # auc_list = np.zeros((epochs))
    model = VAE(image_channels=image_channels).to(device) 
    if iteration !=0:
        model.load_state_dict(torch.load(model_dir+"/pth/WESAD_uaed.pth"))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)  #1e-5  0.001
    
    for epoch in range(epochs):
        for idx, [images,_] in enumerate(train_loader):
            images = images.to(device)
            # print(images.shape)
            recon_images, mu, logvar, z, mu_hat, logvar_hat = model(images)
            if iteration==0: 
                loss, bce, kld = model.loss_fn(recon_images, images, mu, logvar, gmm_mu=None, gmm_var=None, iteration=iteration, mu_hat=mu_hat, logvar_hat=logvar_hat,test=False)
            else:
                loss, bce, kld = model.loss_fn(recon_images, images, mu, logvar, gmm_mu[idx], gmm_var[idx], iteration=iteration, mu_hat=mu_hat, logvar_hat=logvar_hat,test=False)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % 100 ==0:   
                to_print = "Epoch[{}/{}] idx:{} Loss: {:.3f} BCE: {:.3f} KLD: {:.3f}".format(epoch+1, epochs, idx,
                              loss.data/batchsize, bce.data/batchsize, kld.data/batchsize)
                print(to_print)
        loss_list[epoch] = loss
        torch.save(model.state_dict(), model_dir+"/pth/WESAD_uaed.pth")
        # auc = test(0, test_loader , ytest_ECG, model_dir="./model/debug")
        # auc_list[epoch] = auc
    torch.save(model.state_dict(), model_dir+"/pth/WESAD_uaed.pth")
    
    #loss curve
    plt.figure()
    plt.plot(range(0,epochs), loss_list, color="blue", label="loss")
    plt.xlabel('epochs')
    plt.ylabel('loss function')
    plt.savefig('results/UAED_loss_wesad.png')
    plt.close()
    print("loss_list:{}".format(loss_list))
    np.save('results/UAED_loss_wesad.npy', loss_list)
    #auc curve
    # plt.figure()
    # plt.plot(range(0,epochs), auc_list, color="blue", label="auc")
    # plt.xlabel('epochs')
    # plt.ylabel('AUC-ROC')
    # plt.savefig('results/UAED_auc_wesad.png')
    # # plt.close()
    # print("loss_list:{}".format(auc_list))
    # np.save('results/UAED_auc_wesad.npy', auc_list)  
    model.eval()
    for idx, [images,_] in enumerate(all_loader):
        images = images.to(device)
        print('images shape:{} idx:{}'.format(images.shape, idx))
        recon_images, mu, logvar, z_g , mu_hat, logvar_hat= model(images)
        z_g = z_g.cpu()
    return z_g.detach().numpy()  

def test_result(y_test, score):
    threshold = np.linspace(min(score),max(score),200)
    acc_list = []
    f1_list = []
    auc_list = []
    for t in threshold:
        y_pred = (score>t).astype(np.int)
        acc_list.append(accuracy_score(y_pred,y_test))
        f1_list.append(f1_score(y_pred,y_test))
        auc_list.append(roc_auc_score(y_test,y_pred ))
    i = np.argmax(auc_list)   
    t = threshold[i]
    f1 = f1_list[i]
    print('Recommended threshold: %.3f, related f1 score: %.3f'%(t,f1))
    y_pred = (score>t).astype(np.int)
    FN = ((y_test==1) & (y_pred==0)).sum()
    FP = ((y_test==0) & (y_pred==1)).sum()
    TP = ((y_test==1) & (y_pred==1)).sum()
    TN = ((y_test==0) & (y_pred==0)).sum()
    precision, recall, _thresholds = precision_recall_curve(y_test, y_pred)
    area = auc(recall, precision)
    print('\n' + 'StackedTS CNN')
    print('Test set : AUC_ROC: {:.3f} '.format(roc_auc_score(y_test, y_pred))) 
    print('AUC_PR: {:.4f}'.format(area))
    print('precision: {:.4f}'.format(TP/(TP+FP)))
    print('Recall: {:.4f}'.format(TP/(FN+TP)))
    print('F1 score: {:.4f}'.format(f1_score(y_pred,y_test)))
    print('accuracy score: {:.4f}'.format(accuracy_score(y_pred,y_test)))
    print('MIoU:{:.4f}'.format(0.5*(TP/(TP+FP+FN)+TN/(TN+FP+FN))))
    #AUC-ROC curve
    fpr,tpr,threshold = roc_curve(y_test, score) 
    roc_auc = auc(fpr,tpr) 
    linecolors = plt.get_cmap('Set1').colors
    lw = 2
    xyticksize = 16
    xylabelfontsize = 18
    legendfontsize = 12
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color=linecolors[7],
              lw=lw, label='uead\n  AUC_ROC: {:.3f} '.format(roc_auc_score(y_test, y_pred)) ) #(proposed)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tick_params(labelsize=xyticksize)
    plt.xlabel('False Positive Rate', fontsize=xylabelfontsize)
    plt.ylabel('True Positive Rate', fontsize=xylabelfontsize)
    plt.legend(loc="lower right", labelspacing=.5, ncol=2,fontsize=legendfontsize)
    plt.show()
    return roc_auc_score(y_test, y_pred)

def test( iteration, test_loader, label, lam, model_dir):    
    model = VAE(image_channels=1).to(device)   
    model.load_state_dict(torch.load(model_dir+"/pth/WESAD_uaed.pth")) 
    model.eval()
    anomaly_score = []
    with torch.no_grad():
        for idx, [images,_] in enumerate(test_loader):
            images = images.to(device)
            recon_images, mu, logvar ,z , mu_hat, logvar_hat= model(images) 
            loss, bce, kld = model.loss_fn(recon_images, images, mu, logvar, gmm_mu=None, gmm_var=None, iteration=0 , mu_hat=mu_hat, logvar_hat=logvar_hat, test=True)
            w_score = torch.mean((images - mu_hat).pow(2) / torch.exp(logvar_hat))
            p = torch.mean(F.sigmoid(Normal(mu_hat,  torch.exp(0.5 * logvar_hat)).log_prob(images)).squeeze())
            score = -p + lam * w_score
            anomaly_score.append(score.cpu().detach().numpy())
    aucroc = test_result(label, anomaly_score)
    return aucroc


