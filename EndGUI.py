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


def plot_ap():
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.add_patch(Rectangle((-100,-100), 200, 96, alpha = 0.5, color = '#ffcd4d'))
    ax.add_patch(Rectangle((2,-8), 4, 14, alpha = 0.5,color = '#cc8080'))
    ax.add_patch(Rectangle((2,6), 4, 2, alpha = 0.5,color="#4dcc4d"))
    return fig, ax

def plot_ds_model2(fig, ds, target, limits, file_num):
    nx = ny = 50
    axlim = [-8,8,-8,8]
    ax_x = np.linspace(axlim[0], axlim[1], nx)
    ax_y = np.linspace(axlim[2], axlim[3], ny)
    x_tmp, y_tmp = np.meshgrid(ax_x,ax_y)
    x = np.transpose([x_tmp[:], y_tmp[:]])


def plot_ds_model(ds, target, filenum, limits, quality='medium'):
    if quality == 'high':
        nx = 400
        ny = 400
    elif quality == 'medium':
        nx = 200
        ny = 200
    else:
        nx = 50
        ny = 50
    fig,ax = plot_ap()
    axlim = limits
    ax_x = np.linspace(axlim[0], axlim[1], nx)
    ax_y = np.linspace(axlim[2], axlim[3], ny)
    x_tmp, y_tmp = np.meshgrid(ax_x, ax_y)
    x = np.vstack((y_tmp.flatten(),x_tmp.flatten()))
    x_ = x
    xd = ds(x_)
    np.savetxt("lpv_ds"+str(filenum)+".csv", xd, delimiter=',')
    #TODO: Compare the lpv_ds function outputs here:
    #file = os.getcwd()+"/dsltl/experiments/scoop_seed_04/lpv_ds"+str(filenum)+".csv"
    #matlpvds = np.loadtxt(file, delimiter=',')
    #print("matlpvds shape:", matlpvds.shape)
    #print("xd shape:", xd.shape)
    #matlpvds_sorted = np.sort(matlpvds, axis = 1)
    #xd_sorted = np.sort(xd,axis = 1)
    #print("Are arrays equal?", np.isclose(matlpvds_sorted, xd_sorted, rtol = 1e-3, atol = 1e-3))


    h = plt.streamplot(
        x_tmp, y_tmp, xd[0, :].reshape((ny, nx)), xd[1, :].reshape((ny, nx)),
        density=1,
        color='gray',
        linewidth=0.75
    )
    h_att = plt.plot(target[0], target[1], 'o')
    h_att[0].set_markerfacecolor([1, 0.5, 0])
    h_att[0].set_markersize(15)
    h_att[0].set_markeredgecolor([1, 0.5, 0])
    plt.show()
    return h, xd

def lpv_ds(x, ds_gmm, A_g, b_g,filenum):
    #print("x shape: ", filenum)
    N, M = np.shape(x)
    K = len(ds_gmm['Priors'][0][0][0])
    x_ = np.array([x[1,:],x[0,:]])
    beta_k_x = posterior_probs_gmm(x_, ds_gmm, 'norm')
    #print("Beta[0,0", beta_k_x[0,0])
    x_dot = np.zeros((N, M))
    #print("x[:,0]",x[:,0])
    for i in range(M):
        if b_g.shape[1] > 1:
            f_g = np.zeros((N, K))
            for k in range(K):
                x_p = [x[1,i], x[0,i]]
                #f_g[:,k] = np.multiply(beta_k_x[k,i], (np.matmul(A_g[:,:,k], x[:,i])) + b_g[:,k])
                f_g[:,k] = np.multiply(beta_k_x[k,i], (np.matmul(A_g[:,:,k], x_p)) + b_g[:,k])
                #f_g[:,k] = np.matmul(A_g[:,:,k], x[:,i])
                #if(i==0 and k == 0):
                #    print("beginning")
                #    print(x_p)
                #    print(A_g[:,:,k])
                #    print(np.matmul(A_g[:,:,k], x_p))
                #    print("beta", beta_k_x[k,i])
                #    print((np.matmul(A_g[:,:,k], x_p)) + b_g[:,k])
            f_g = np.sum(f_g, axis=1)
        else:
            f_g = A_g @ x[:, i] + b_g
        x_dot[:, i] = f_g
    #print("fg: ", x_dot)

    file = os.getcwd()+"/dsltl/experiments/scoop_seed_04/post_p/x_dot"+str(3-filenum)+".csv"
    data = np.loadtxt(file, delimiter=',')
    data_sorted = np.sort(data, axis = 1)
    x_dot_sorted = np.sort(x_dot,axis = 1)
    #print("data sorted" , data_sorted)
    #print("x dot sorterd: ", x_dot_sorted)
    print("Are arrays equal?", np.isclose(data_sorted, x_dot_sorted, rtol = 1e-2, atol = 1e-2))

    return x_dot

def posterior_probs_gmm(x, gmm, type):
    N, M = np.shape(x)
    Mu = np.array(gmm['Mu'])[0][0]
    Priors = gmm['Priors'][0][0][0]
    Sigma = gmm['Sigma'][0][0]
    #print("priros: ", Priors.shape)
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
    if type == 'norm':
        #Pk_x = alpha_Px_k / np.sum(alpha_Px_k, axis=0, keepdims=True)
        sums = np.sum(alpha_Px_k, axis=0, keepdims=True)
        sums_repeated = np.tile(sums, (K,1))
        Pk_x = alpha_Px_k / sums_repeated
    elif type == 'un-norm':
        Pk_x = alpha_Px_k
    else:
        raise ValueError("Invalid type. Must be 'norm' or 'un-norm'.")

    return Pk_x


def ml_gaussPDF(Data, Mu, Sigma):
    # (D x N) - repmat((D x 1),1,N)
    # (D x N) - (D x N)
    nbVar, nbData = Data.shape
    Mu = np.reshape(Mu,(Mu.shape[0],1))
    Mus = np.tile(Mu, (1, nbData))
    Data = (Data - Mus).T
    # Data = (N x D)
    
    # (N x 1)
    prob = np.sum((Data @ np.linalg.inv(Sigma)) * Data, axis=1)
    prob = np.exp(-0.5 * prob) / np.sqrt((2 * np.pi) ** nbVar * abs(np.linalg.det(Sigma) + np.finfo(float).tiny)) + np.finfo(float).tiny
    return prob

models = []
limits = [-8,8,-8,8]
directory = os.getcwd()+"/dsltl/experiments/scoop_seed_04/"
files = glob.glob(directory + "/ds0*")

i = 0
for f in files:
    filename = directory + os.path.basename(f)
    print(filename)
    ds = loadmat(filename)
    #print("ds gmm mu: ", ds['ds_gmm']["Mu"])
    #print("ds A_k", ds["A_k"])
    #print("ds bk", ds["b_k"])
    ds["ds_lpv"] = lambda x: lpv_ds(x, ds["ds_gmm"], ds["A_k"], ds["b_k"],i)
    models.append(ds)
    hs = plot_ds_model(ds["ds_lpv"], ds["att"], i, limits,'low')
    i+=1