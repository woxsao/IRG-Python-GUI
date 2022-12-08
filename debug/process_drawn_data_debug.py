from scipy.signal import savgol_filter
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

def process_drawn_data(derivs_list):
    data = {"drawn_data":[], "Data":[], "Data_sh":[], "att":[], "x0_all":[], "dt": []}
    data = [{"drawn_data":[]}]
    att_list = []
    for traj in derivs_list:
        att = np.row_stack(traj[:2, -1])
        att_list.append(att)
    att_list = np.column_stack(att_list)

    att = np.mean(att_list, axis = 1)
    att = np.reshape(att, (2,1))
    shifts = att_list-repmat(att, 1, att_list.shape[1])
    #print("shifts:", shifts)
    #print("att:", att)
    #print("att list: ", att_list)
    Data = np.empty((4,0))
    x0_all = []
    Data_sh = np.empty((4,0))
    num_traj = len(derivs_list)
    for i in range(len(derivs_list)):
        data_ = derivs_list[i].copy()
        #print('first col in process data:', data_[:2,0])
        s = np.row_stack(shifts[:,i])
        shifts_ = repmat(s, 1, data_.shape[1])
        #print("shifts", shifts_[:,100])
        data_[:2,:] = data_[:2,:]-shifts_
        #print("data after shift", data_[:2,4])
        data_[2:, -1] = np.zeros((2))
        #print(data_[2:,-1])
        data_[2:, -2] = (data_[2:,-1] + np.zeros((2)))/2
        data_[2:, -3] = (data_[2:,-3] + data_[2:,-2])/2
        #print(data_[:,-3:])
        Data = np.column_stack((Data, data_.tolist()))
        #print(data_[:2,0])
        x0_all.append((data_[:2, 0]).copy())
        #print(x0_all)
        data_[:2, :] = data_[:2, :]-repmat(att, 1, data_.shape[1])
        data_[2:4, -1] = np.zeros((2))
        Data_sh =  np.column_stack((Data_sh, data_.tolist()))
        #derivs_list[i] = data_
    data_12 = derivs_list[0][:,:2]
    dt = abs((data_12[0,0] - data_12[0,1])/data_12[2,0])
    #print("dt",dt)
    for i in range(len(derivs_list)): 
        derivs_list[i] = derivs_list[i].tolist()

    #derivs_list  = np.array(derivs_list)
    #print('drawn data before: ', len(derivs_list[0]))
    #print(x0_all)
    derivs_list = np.transpose(derivs_list)
    #print('drawn data shape: ', len(derivs_list))
    #print("Data shape", Data.shape)
    data[0]["drawn_data"] = derivs_list
    data[0]["Data"] = Data
    data[0]["Data_sh"] = Data_sh
    data[0]["att"] = att
    data[0]["x0_all"] = np.transpose(np.array(x0_all))
    data[0]["dt"] = dt
    data[0]["num_traj"] = num_traj
    return_dict = {'seg':data}
    return return_dict

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
    for dim in range(x.shape[1]):
        y = np.transpose(x[:, dim])
        half_win = ((window_size+1)//2)-1
        ysize = y.shape[0]
        for n in range((window_size+1)//2,ysize-(window_size+1)//2+1):
            #print("n:",n)
            for dx_order in range(0,nth_order+1):
                d_nth_x[n, dim, dx_order+1] = np.dot(y[n-half_win-1:n+half_win], (factorial(dx_order)/(dt**dx_order))*g[:,dx_order])
                #print(foo)
    crop_size = (window_size+1)//2
    end_crop = d_nth_x.shape[0]-crop_size
    
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
derivs_list = []
for traj in traj_list:
    x = traj[0,:]
    y = traj[1,:]
    t = traj[2,:]
    dt = np.mean(np.diff(t))
    x_obs = np.transpose(traj[:2,:])
    dx_nth = sgolay_time_derivatives(x_obs, dt, 2,3,15)
    drawn_data = np.row_stack((np.transpose(dx_nth[:,:,1]), np.transpose(dx_nth[:,:,2])))
    print(drawn_data[:,:3])
    derivs_list.append(drawn_data)


#print("First col", derivs_list[1][:2,0])
seg = process_drawn_data(derivs_list)["seg"][0]
for key, val in seg.items():
    if key not in set(["drawn_data", "Data", "Data_sh"]):
        print(key, val)
    else:
        print(key, val[:,:5])

