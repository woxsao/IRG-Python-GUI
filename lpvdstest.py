from matplotlib.patches import Rectangle
import numpy as np
import matplotlib.pyplot as plt
import time 
from matplotlib.widgets import Button
import os
import glob
import csv
from scipy.interpolate import interp2d
from scipy.interpolate  import griddata
from scipy.io import loadmat
import math
import sys
from scipy.stats import multivariate_normal
def ml_gaussPDF(Data, Mu, Sigma):
    print("Raw data", Data[:,:6])
    print('data shape', Data.shape)
    print("mu shape", Mu.shape)
    print("Sigma shape", Sigma.shape)
    # (D x N) - repmat((D x 1),1,N)
    # (D x N) - (D x N)
    nbVar, nbData = Data.shape
    Mu = np.reshape(Mu,(Mu.shape[0],1))
    Mus = np.tile(Mu, (1, nbData))
    print("Mu, ", Mu)
    Data = (Data - Mus).T
    print("Data:", Data[:6,:])
    # Data = (N x D)
    
    # (N x 1)
    prob = np.sum((Data @ np.linalg.inv(Sigma)) * Data, axis=1)
    print("Sigma!: ", Sigma)
    print("prob before math", prob[99:106])
    print("tiny number", np.finfo(float).tiny)
    prob = np.exp(-0.5 * prob) / np.sqrt((2 * np.pi) ** nbVar * abs(np.linalg.det(Sigma) + np.finfo(float).tiny)) + np.finfo(float).tiny
    return prob

def posterior_probs_gmm(x, gmm, type):
    N, M = np.shape(x)
    Mu = np.array(gmm['Mu'])[0][0]
    Priors = gmm['Priors'][0][0][0]
    Sigma = gmm['Sigma'][0][0]
    print("priros: ", Priors.shape)
    #print(len(Priors))
    K = len(Priors)
    # Compute mixing weights for multiple dynamics
    Px_k = np.zeros((K, M))
    # Compute probabilities p(x^i|k)
    for k in range(K):
        #Px_k[k,:] = np.apply_along_axis(lambda x_i: multivariate_normal.pdf(x_i, mean=Mu[:,k], cov=Sigma[:,:,k]), axis=0, arr=x)+eps
        #Px_k[k,:] = np.apply_along_axis(lambda x_i: multivariate_normal.pdf(x_i, mean=Mu[:,k], cov=Sigma[:,:,k]), axis=0, arr=x) + np.finfo(np.longdouble).eps
        Px_k[k,:] = ml_gaussPDF(x,Mu[:,k],Sigma[:,:,k])+np.spacing(1)
    # Compute posterior probabilities p(k|x)
    alpha_Px_k = np.tile(Priors.T, (M,1)).T * Px_k
    print("alphaPx", alpha_Px_k[:,6])
    print("priors:", Priors)
    #print("pre alpha:",np.tile(Priors.T, (M,1)).T)
    print("normal Pkx", Px_k[:,6])
    if type == 'norm':
        print("repmat end", np.sum(alpha_Px_k, axis=0, keepdims=True)[:,:6])
        #Pk_x = alpha_Px_k / np.sum(alpha_Px_k, axis=0, keepdims=True)
        sums = np.sum(alpha_Px_k, axis=0, keepdims=True)
        sums_repeated = np.tile(sums, (K,1))
        Pk_x = alpha_Px_k / sums_repeated
    elif type == 'un-norm':
        Pk_x = alpha_Px_k
    else:
        raise ValueError("Invalid type. Must be 'norm' or 'un-norm'.")

    print("returning", Pk_x[:,6:10])
    return Pk_x
def lpv_ds(x, ds_gmm, A_g, b_g,filenum):
    print("x shape: ", filenum)
    N, M = np.shape(x)
    K = len(ds_gmm['Priors'][0][0][0])
    print("K = ", K)
    print("Priros:", ds_gmm["Priors"])
    beta_k_x = posterior_probs_gmm(x, ds_gmm, 'norm')
    print("Beta", beta_k_x[:,:6])
    x_dot = np.zeros((N, M))
    print("x[:,0]",x[:,0])
    for i in range(M):
        if b_g.shape[1] > 1:
            f_g = np.zeros((N, K))
            for k in range(K):
                x_p = [x[0,i], x[1,i]]
                #f_g[:,k] = np.multiply(beta_k_x[k,i], (np.matmul(A_g[:,:,k], x[:,i])) + b_g[:,k])
                f_g[:,k] = np.multiply(beta_k_x[k,i], (np.matmul(A_g[:,:,k], x_p)) + b_g[:,k])
                #f_g[:,k] = np.matmul(A_g[:,:,k], x[:,i])
                if(i==0 and k == 1):
                    print("HIHIHIHIHIHIHIHIHIHIHIHIHIHIHIHIHIHI")
                    print(x_p)
                    print(A_g[:,:,k])
                    print(np.matmul(A_g[:,:,k], x_p))
                    print("b_g k", b_g[:,k])
                    print("beta", beta_k_x[k,i])
                    print("np matmul line", (np.matmul(A_g[:,:,k], x_p)) + b_g[:,k])
            f_g = np.sum(f_g, axis=1)
            if(i==0):
                print("f_g sum", f_g)
        else:
            f_g = A_g @ x[:, i] + b_g
        x_dot[:, i] = f_g
    print("fg: ", x_dot)

    file = os.getcwd()+"/dsltl/experiments/scoop_seed_04/post_p/x_dot"+str(3-filenum)+".csv"
    data = np.loadtxt(file, delimiter=',')
    print("data sorted" , data)
    print("x dot sorterd: ", x_dot)

    return x_dot

models = []
limits = [-8,8,-8,8]
directory = os.getcwd()+"/dsltl/experiments/scoop_seed_04/"
files = glob.glob(directory + "/ds01*")

i = 0
for f in files:
    nx = ny = 50
    axlim = limits
    ax_x = np.linspace(axlim[0], axlim[1], nx)
    ax_y = np.linspace(axlim[2], axlim[3], ny)
    x_tmp, y_tmp = np.meshgrid(ax_x, ax_y)
    x = np.vstack((x_tmp.flatten(), y_tmp.flatten()))
    filename = directory + os.path.basename(f)
    print(filename)
    ds = loadmat(filename)
    print("ds gmm mu: ", ds['ds_gmm']["Mu"])
    print("ds A_k", ds["A_k"])
    print("ds bk", ds["b_k"])
    res = lpv_ds(x,ds['ds_gmm'],ds["A_k"],ds["b_k"],2)
    print(res[:,:6])
    