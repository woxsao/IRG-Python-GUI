from scipy.io import savemat
from matplotlib.patches import Rectangle
import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt
import time 
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
    


start = time.time()
times = []
fig,ax = plot_ap()
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
    traj_list.append(trajectory)

def clear_data(event):
    if event.key=='c':
        traj_list.clear()
    if event.key == 'd':
        global draw_traj
        draw_traj = False
        plt.close()

def sgolay_time_derivatives(x, dt, nth_order, n_polynomial, window_size):
    """
    This function applies a 2d filter over the data according to the sgolay derivative specification
    TODO: 
    rewrite the function so that not hard coding the kernel, hence why n_polynomial is an unused parameter
    """
    if(x.shape[0] < window_size):
        raise ValueError("Window size is too big for data")
    g = pd.read_csv(os.getcwd()+"/IRG-Python-GUI-main/sgolay.csv", header = None).to_numpy()
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

def segment_data(derivs_list):
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
        fig, ax = plot_ap()
        plt.scatter(segments[i][0,:], segments[i][1,:], s = 0.5)
        plt.show()
    plt.show()
    return derivs_list_segment


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
    Data = np.empty((4,0))
    x0_all = []
    Data_sh = np.empty((4,0))
    num_traj = len(derivs_list)
    for i in range(len(derivs_list)):
        data_ = derivs_list[i].copy()
        s = np.row_stack(shifts[:,i])
        shifts_ = repmat(s, 1, data_.shape[1])
        data_[:2,:] = data_[:2,:]-shifts_
        data_[2:, -1] = np.zeros((2))
        data_[2:, -2] = (data_[2:,-1] + np.zeros((2)))/2
        data_[2:, -3] = (data_[2:,-3] + data_[2:,-2])/2
        Data = np.column_stack((Data, data_.tolist()))
        x0_all.append((data_[:2, 0]).copy())
        data_[:2, :] = data_[:2, :]-repmat(att, 1, data_.shape[1])
        data_[2:4, -1] = np.zeros((2))
        Data_sh =  np.column_stack((Data_sh, data_.tolist()))
    data_12 = derivs_list[0][:,:2]
    dt = abs((data_12[0,0] - data_12[0,1])/data_12[2,0])
    for i in range(len(derivs_list)): 
        derivs_list[i] = derivs_list[i].tolist()

    derivs_list = np.transpose(derivs_list)
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
#This section calculates the derivatives 
derivs_list = []
for traj in traj_list:
    x_obs = np.transpose(traj[:2,:])
    t = traj[2,:]
    dt = np.mean(np.diff(t))
    dx_nth = sgolay_time_derivatives(x_obs, dt, 2,3,15)
    traj_drawn = np.row_stack((np.transpose(dx_nth[:,:,1]), np.transpose(dx_nth[:,:,2])))
    derivs_list.append(traj_drawn)


seg_list = segment_data(derivs_list)
trajmat = process_drawn_data(derivs_list)
directory = os.getcwd()+"/dsltl/experiments/scoop_seed_04"

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