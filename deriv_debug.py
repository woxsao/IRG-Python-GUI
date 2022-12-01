from scipy.signal import savgol_filter, savgol_coeffs
from scipy.io import savemat, loadmat
from matplotlib.patches import Rectangle
import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt
import time 
from matplotlib.widgets import Button
import os
import glob
import pandas as pd
from math import factorial


def sgolay_time_derivatives(x, dt, nth_order, n_polynomial, window_size):
    if(x.shape[0] < window_size):
        raise ValueError("Window size is too big for data")
    g = pd.read_csv("/Users/MonicaChan/Desktop/UROP/Python Implementation/sgolay.csv", header = None).to_numpy()
    #g = np.array(savgol_coeffs(window_size, 3))
    d_nth_x = np.empty((x.shape[0],x.shape[1],nth_order+2))
    print("shape x:", x.shape)
    for dim in range(x.shape[1]):
        y = np.transpose(x[:, dim])
        half_win = ((window_size+1)//2)-1
        ysize = y.shape[0]
        print("half win", half_win)
        print("ysize:", ysize)
        for n in range((window_size+1)//2,ysize-(window_size+1)//2+1):
            #print("n:",n)
            for dx_order in range(0,nth_order+1):
                print("dx order:", dx_order)
                print('factorial coeff:',y[n-half_win:n+half_win+1])
                d_nth_x[n, dim, dx_order+1] = np.dot(y[n-half_win-1:n+half_win], (factorial(dx_order)/(dt**dx_order))*g[:,dx_order])
                #print(foo)
    crop_size = (window_size+1)//2
    end_crop = d_nth_x.shape[0]-crop_size
    
    print(end_crop)
    d_nth_x = d_nth_x[crop_size:end_crop+1,:,:]
    #print(d_nth_x)
    return d_nth_x
directory = "/Users/MonicaChan/Desktop/UROP/dsltl/experiments/scoop_seed_04/"
trajectories = glob.glob(directory+"dem*")
traj_list = []
for f in trajectories:
    data = pd.read_csv(f, header = None).to_numpy()
    traj_list.append(data)
#mat = loadmat(directory + "traj.mat")
#
#drawn_data = mat["seg"][0][0][0][0]
#for traj in drawn_data:
#    traj = np.array(traj)
#    xy = traj[:2,:]
#    traj_list.append(xy)

derivs_list = []
drawn_data = np.empty((4,1))
files = glob.glob('/Users/MonicaChan/Desktop/UROP/Python Implementation/trajectory_data/*')
for f in files:
    os.remove(f)
for traj in traj_list:
    x = traj[0,:]
    y = traj[1,:]
    t = traj[2,:]
    dt = np.mean(np.diff(t))
    x_obs = np.transpose(traj[:2,:])
    dx_nth = sgolay_time_derivatives(x_obs, dt, 2,3,15)
    drawn_data = np.row_stack((np.transpose(dx_nth[:,:,1]), np.transpose(dx_nth[:,:,2])))
    print(drawn_data[:,:4])
        #print(dx[:,0])
    #dy = np.array(savgol_filter(y, window_length = 15, polyorder = 3, deriv = 2, delta = dt))
    #traj = np.vstack((traj[0:2, :], dx,dy))
    #derivs_list.append(traj)
    #DIR = '/Users/MonicaChan/Desktop/UROP/Python Implementation/trajectory_data'
    #traj_num =  len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    #filename = "/Users/MonicaChan/Desktop/UROP/Python Implementation/trajectory_data/deriv" + str(traj_num) + ".csv"
    #np.savetxt(filename,traj,delimiter=',')