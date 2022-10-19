from matplotlib.patches import Rectangle
import numpy as np
import matplotlib.pyplot as plt
import time 
from matplotlib.widgets import Button
import os
import glob

files = glob.glob('/Users/MonicaChan/Desktop/UROP/Python Implementation/trajectory_data/*')
for f in files:
    os.remove(f)

start = time.time()
times = []
plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)
ax.add_patch(Rectangle((-100,-100), 200, 96, alpha = 0.5, color = '#ffcd4d'))
ax.add_patch(Rectangle((2,-8), 4, 14, alpha = 0.5,color = '#cc8080'))
ax.add_patch(Rectangle((2,6), 4, 2, alpha = 0.5,color="#4dcc4d"))
x = []
y = []
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
    print(trajectory)
    print(trajectory.shape)
    DIR = '/Users/MonicaChan/Desktop/UROP/Python Implementation/trajectory_data'
    traj_num =  len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    filename = "/Users/MonicaChan/Desktop/UROP/Python Implementation/trajectory_data/trajectory" + str(traj_num) + ".csv"
    np.savetxt(filename,trajectory)

def clear_data(event):
    if event.key=='c':
        files = glob.glob('/Users/MonicaChan/Desktop/UROP/Python Implementation/trajectory_data/*')
        for f in files:
            os.remove(f)

for j in range(10000):
    fig.canvas.restore_region(bg)
    cid = fig.canvas.mpl_connect('motion_notify_event', onclick)
    cid = fig.canvas.mpl_connect('button_release_event', offclick)
    cid = fig.canvas.mpl_connect('key_press_event', clear_data)
    pts.set_ydata(y)
    pts.set_xdata(x)
    ax.draw_artist(pts)
    #ax.scatter(x,y, c = 'red')
    #plt.scatter(x,y, c = 'red')
    #fig.canvas.draw()
    fig.canvas.blit(fig.bbox)
    # flush any pending GUI events, re-painting the screen if needed
    fig.canvas.flush_events()
    plt.pause(0.01)