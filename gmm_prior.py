# -*- coding: utf-8 -*-
"""
@Project: AEDCoder20211214
@File:    UAED
@Author:  Jiaqi
@Description: GMM module, 
Gaussian mixture distribution prior in the VAE's latent space
The implementation of GMM is based on https://www.anarchive-beta.com/entry/2020/11/28/210948
"""

import numpy as np
from scipy.stats import multivariate_normal, wishart, dirichlet 
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score as ARI
from tool import calc_ari
import torch

def train(iteration, x_d,  K, epoch=100):    
    print("GMM Training Start")
    D = len(x_d) 
    dim = len(x_d[0]) 

    #Parameters of pre-distribution
    beta = 0.8; m_d = np.repeat(0.0, dim)
    w_dd = np.identity(dim) * 0.55; nu = dim 
    alpha_k = np.repeat(0.3, K)

    #\mu, \lambda, \pi initialization
    mu_kd = np.empty((K, dim)); lambda_kdd = np.empty((K, dim, dim))
    for k in range(K):
        lambda_kdd[k] = wishart.rvs(df=nu, scale=w_dd, size=1)
        mu_kd[k] = np.random.multivariate_normal(mean=m_d, cov=np.linalg.inv(beta * lambda_kdd[k])).flatten()
    pi_k = dirichlet.rvs(alpha=alpha_k, size=1).flatten()

    # parameter initialization
    eta_dk = np.zeros((D, K))
    z_dk = np.zeros((D, K))
    beta_hat_k = np.zeros(K)
    m_hat_kd = np.zeros((K, dim))
    w_hat_kdd = np.zeros((K, dim, dim))
    nu_hat_k = np.zeros(K)
    alpha_hat_k = np.zeros(K)

    trace_s_in = [np.repeat(np.nan, D)]
    trace_mu_ikd = [mu_kd.copy()]
    trace_lambda_ikdd = [lambda_kdd.copy()]
    trace_pi_ik = [pi_k.copy()]
    trace_beta_ik = [np.repeat(beta, K)]
    trace_m_ikd = [np.repeat(m_d.reshape((1, dim)), K, axis=0)]
    trace_w_ikdd = [np.repeat(w_dd.reshape((1, dim, dim)), K, axis=0)]
    trace_nu_ik = [np.repeat(nu, K)]
    trace_alpha_ik = [alpha_k.copy()]
    ARI = np.zeros((epoch))
    max_ARI = 0 

    for i in range(epoch):
        pred_label = []
        
        # Sampling of z
        for k in range(K):
            tmp_eta_n = np.diag(
                -0.5 * (x_d - mu_kd[k]).dot(lambda_kdd[k]).dot((x_d - mu_kd[k]).T)
            ).copy() 
            tmp_eta_n += 0.5 * np.log(np.linalg.det(lambda_kdd[k]) + 1e-7)
            tmp_eta_n += np.log(pi_k[k] + 1e-7)
            eta_dk[:, k] = np.exp(tmp_eta_n)
        eta_dk /= np.sum(eta_dk, axis=1, keepdims=True) 
        
        # Sampling of latent variable
        for d in range(D):
            z_dk[d] = np.random.multinomial(n=1, pvals=eta_dk[d], size=1).flatten()
            pred_label.append(np.argmax(z_dk[d]))
            
        #Sampling of \mu， \lambda
        for k in range(K):
            # Calculate the posterior distribution parameters of mu
            beta_hat_k[k] = np.sum(z_dk[:, k]) + beta
            m_hat_kd[k] = np.sum(z_dk[:, k] * x_d.T, axis=1)
            m_hat_kd[k] += beta * m_d
            m_hat_kd[k] /= beta_hat_k[k]
            
            # Calculate the posterior distribution parameters of lambda
            tmp_w_dd = np.dot((z_dk[:, k] * x_d.T), x_d)
            tmp_w_dd += beta * np.dot(m_d.reshape(dim, 1), m_d.reshape(1, dim))
            tmp_w_dd -= beta_hat_k[k] * np.dot(m_hat_kd[k].reshape(dim, 1), m_hat_kd[k].reshape(1, dim))
            tmp_w_dd += np.linalg.inv(w_dd)
            w_hat_kdd[k] = np.linalg.inv(tmp_w_dd)
            nu_hat_k[k] = np.sum(z_dk[:, k]) + nu
            
            lambda_kdd[k] = wishart.rvs(size=1, df=nu_hat_k[k], scale=w_hat_kdd[k])
            mu_kd[k] = np.random.multivariate_normal(
                mean=m_hat_kd[k], cov=np.linalg.inv(beta_hat_k[k] * lambda_kdd[k]), size=1
            ).flatten()

        # Sampling of \pi
        alpha_hat_k = np.sum(z_dk, axis=0) + alpha_k
        pi_k = dirichlet.rvs(size=1, alpha=alpha_hat_k).flatten()
        
        ARI[i] = np.round(calc_ari(pred_label,1)[0],3)
        if max_ARI <= ARI[i]:
            max_ARI = ARI[i]
        
        if i == 0 or (i+1) % 50 == 0:
            print(f"====> Epoch: {i+1}, ARI: {ARI[i]}, MaxARI: {max_ARI}")

        _, z_n = np.where(z_dk == 1)
        trace_s_in.append(z_n.copy())
        trace_mu_ikd.append(mu_kd.copy())
        trace_lambda_ikdd.append(lambda_kdd.copy())
        trace_pi_ik.append(pi_k.copy())
        trace_beta_ik.append(beta_hat_k.copy())
        trace_m_ikd.append(m_hat_kd.copy())
        trace_w_ikdd.append(w_hat_kdd.copy())
        trace_nu_ik.append(nu_hat_k.copy())
        trace_alpha_ik.append(alpha_hat_k.copy())

    mu_d = np.zeros((D,dim)) 
    var_d = np.zeros((D,dim)) 
    for d in range(D):
        var_d[d] = np.diag(np.linalg.inv(lambda_kdd[pred_label[d]]))
        mu_d[d] = mu_kd[pred_label[d]]

    return mu_d, var_d , max_ARI

