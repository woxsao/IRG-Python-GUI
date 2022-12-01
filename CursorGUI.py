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
from math import factorial

def plot_ap():
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.add_patch(Rectangle((-100,-100), 200, 96, alpha = 0.5, color = '#ffcd4d'))
    ax.add_patch(Rectangle((2,-8), 4, 14, alpha = 0.5,color = '#cc8080'))
    ax.add_patch(Rectangle((2,6), 4, 2, alpha = 0.5,color="#4dcc4d"))
    return fig, ax
    
files = glob.glob('/Users/MonicaChan/Desktop/UROP/Python Implementation/trajectory_data/*')
for f in files:
    os.remove(f)

start = time.time()
times = []
#plt.ion()
#fig, ax = plt.subplots(figsize=(3.5, 3.5))
#ax.set_xlim(-8, 8)
#ax.set_ylim(-8, 8)
#ax.add_patch(Rectangle((-100,-100), 200, 96, alpha = 0.5, color = '#ffcd4d'))
#ax.add_patch(Rectangle((2,-8), 4, 14, alpha = 0.5,color = '#cc8080'))
#ax.add_patch(Rectangle((2,6), 4, 2, alpha = 0.5,color="#4dcc4d"))
fig,ax = plot_ap()
rectangles = [np.array([-100,-100,100,-4]), np.array([2,-8,6,6]), np.array([2,6,6,8])]
x = []
y = []
traj_list = []
traj_deriv_list = []
traj_num = 1
(pts,) = ax.plot(x, y, animated = True)
data = []
plt.show(block = False)
plt.pause(0.1)

bg = fig.canvas.copy_from_bbox(fig.bbox)
ax.draw_artist(pts)
fig.canvas.blit(fig.bbox)

# Add UI buttons for data/figure manipulation
store_btn  = plt.axes([0.67, 0.05, 0.075, 0.05])
clear_btn  = plt.axes([0.78, 0.05, 0.075, 0.05])
bclear     = Button(clear_btn, 'clear')


draw_traj= True
shown_plot = False

def onclick(event):
    if event.button == 1:
        x.append(event.xdata)
        y.append(event.ydata)
        cur = time.time()-start
        times.append(cur)
def offclick(event):
    trajectory = np.array((x,y))
    x.clear()
    y.clear()
    trajectory = np.vstack([trajectory, times])
    times.clear()
    #print(trajectory)
    #print(trajectory.shape)
    DIR = '/Users/MonicaChan/Desktop/UROP/Python Implementation/trajectory_data'
    traj_num =  len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    filename = "/Users/MonicaChan/Desktop/UROP/Python Implementation/trajectory_data/trajectory" + str(traj_num) + ".csv"
    np.savetxt(filename,trajectory, delimiter = ',')
    traj_list.append(trajectory)

def clear_data(event):
    if event.key=='c':
        files = glob.glob('/Users/MonicaChan/Desktop/UROP/Python Implementation/trajectory_data/*')
        for f in files:
            os.remove(f)
        traj_list.clear()
    if event.key == 'd':
        global draw_traj
        draw_traj = False
        plt.close()
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
            for dx_order in range(0,nth_order+1):
                d_nth_x[n, dim, dx_order+1] = np.dot(y[n-half_win-1:n+half_win], (factorial(dx_order)/(dt**dx_order))*g[:,dx_order])
 
    crop_size = (window_size+1)//2
    end_crop = d_nth_x.shape[0]-crop_size
    
    print(end_crop)
    d_nth_x = d_nth_x[crop_size:end_crop+1,:,:]
    return d_nth_x

def segment_data(derivs_list, rectangles):
    sections = [np.array([-100,-4,2,8]), np.array([-100,-100,2,-4]),np.array([2,-8,6,6])]
    segments = [np.empty((4,0)), np.empty((4,0)), np.empty((4,0)),np.empty((4,0))]
    derivs_list_segment = []
    for traj in derivs_list:
        pts = traj[:2,:]
        print(pts.shape)
        for i in range(len(sections)):
            print(len(derivs_list_segment))
            if len(derivs_list_segment) < i+1:
                derivs_list_segment.append(list())
            rect = sections[i]
            ll = np.row_stack((rect[0], rect[1]))
            ur = np.row_stack((rect[2], rect[3]))
            seg = np.all(np.logical_and(ll <= pts, pts < ur), axis=0)
            inroi = traj[:, seg]
            segments[i] = np.column_stack((segments[i], inroi))
            derivs_list_segment[i].append(inroi)
    for i in range(len(segments)):
        #derivs_list_segment.append([segments[i]])
        filename = "/Users/MonicaChan/Desktop/UROP/Python Implementation/segments/segment" + str(i)+".csv"
        np.savetxt(filename, segments[i], delimiter=",")
        fig, ax = plot_ap()
        plt.scatter(segments[i][0,:], segments[i][1,:], s = 0.5)
        plt.show()
    plt.show()
    #derivs_list_segment = np.array(derivs_list_segment)
    #print('derivs list shape: ', derivs_list_segment.shape)
    return derivs_list_segment

def process_drawn_data2(derivs_list):
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
    Data = np.empty((4,1))
    x0_all = []
    Data_sh = np.empty((4,1))
    num_traj = len(derivs_list)
    for i in range(len(derivs_list)):
        data_ = derivs_list[i]
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
        derivs_list[i] = data_
    data_12 = derivs_list[0][:,:2]
    print(data_12)
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
    
while draw_traj == True:
    fig.canvas.restore_region(bg)
    cid = fig.canvas.mpl_connect('motion_notify_event', onclick)
    cid = fig.canvas.mpl_connect('button_release_event', offclick)
    cid = fig.canvas.mpl_connect('key_press_event', clear_data)
    pts.set_ydata(y)
    pts.set_xdata(x)
    ax.draw_artist(pts)
    fig.canvas.blit(fig.bbox)
    # flush any pending GUI events, re-painting the screen if needed
    fig.canvas.flush_events()
    plt.pause(0.01)

#This section plots out for UI purposes:
fig, ax = plt.subplots(figsize=(3, 3))
ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)
ax.add_patch(Rectangle((-100,-100), 200, 96, alpha = 0.5, color = '#ffcd4d'))
ax.add_patch(Rectangle((2,-8), 4, 14, alpha = 0.5,color = '#cc8080'))
ax.add_patch(Rectangle((2,6), 4, 2, alpha = 0.5,color="#4dcc4d"))
for traj in traj_list:
    x = traj[0,:]
    y = traj[1,:]
    plt.plot(x,y)

plt.show()
#plt.pause(5)
#This section calculates the derivatives 
derivs_list = []
for traj in traj_list:
    x_obs = np.transpose(traj[:2,:])
    t = traj[2,:]
    dt = np.mean(np.diff(t))
    dx_nth = sgolay_time_derivatives(x_obs, dt, 2,3,15)
    traj_drawn = np.row_stack((np.transpose(dx_nth[:,:,1]), np.transpose(dx_nth[:,:,2])))
    derivs_list.append(traj_drawn)
    #dx = np.array(savgol_filter(x, window_length = 15, polyorder = 3, deriv = 2, delta = dt))
    #dy = np.array(savgol_filter(y, window_length = 15, polyorder = 3, deriv = 2, delta = dt))
    #traj = np.vstack((traj[0:2, :], dx,dy))
    #derivs_list.append(traj)
    #DIR = '/Users/MonicaChan/Desktop/UROP/Python Implementation/trajectory_data'
    #traj_num =  len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    #filename = "/Users/MonicaChan/Desktop/UROP/Python Implementation/trajectory_data/deriv" + str(traj_num) + ".csv"
    #np.savetxt(filename,traj,delimiter=',')

seg_list = segment_data(derivs_list, rectangles)
trajmat = process_drawn_data(derivs_list)
directory = "/Users/MonicaChan/Desktop/UROP/dsltl/experiments/scoop_seed_04"
filename = "/traj.mat"
files = glob.glob(directory + "/*")
for f in files:
    os.unlink(f)
savemat(directory+filename, trajmat, do_compression = True, oned_as = "column")
for i, seg in enumerate(seg_list):
    filename = "/seg0"+str(i+1)+".mat"
    s = process_drawn_data(seg)
    fig,ax = plot_ap()
    plt.scatter(s['seg'][0]["att"][0],s['seg'][0]["att"][1])
    plt.show()
    savemat(directory+filename, s, do_compression = True, oned_as = "column")