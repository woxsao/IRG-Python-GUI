from logging import exception
from matplotlib.widgets import Cursor
import numpy as np
import matplotlib.pyplot as plt
import time 


start = time.time()
times = []
plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)
x = []
y = []
(pts,) = ax.plot(x, y, animated = True)

plt.show(block = False)
plt.pause(0.1)

bg = fig.canvas.copy_from_bbox(fig.bbox)
ax.draw_artist(pts)
fig.canvas.blit(fig.bbox)
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
    print(trajectory)
    print(trajectory.shape)
    np.savetxt("trajectory.csv", trajectory, delimiter=",")
for j in range(10000):
    fig.canvas.restore_region(bg)
    cid = fig.canvas.mpl_connect('motion_notify_event', onclick)
    cid = fig.canvas.mpl_connect('button_release_event', offclick)
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