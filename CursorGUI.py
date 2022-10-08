from matplotlib.widgets import Cursor
import numpy as np
import matplotlib.pyplot as plt
import time 

def onclick(event):
    if event.button == 1:
        x.append(event.xdata)
        y.append(event.ydata) 

plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))
x = []
y = []
#ax.set_xlim(-8, 8)
#ax.set_ylim(-8, 8)
#cursor = Cursor(ax, useblit=True, color='red', linewidth=2)

for i in range(4000):
    cid = fig.canvas.mpl_connect('motion_notify_event', onclick)
    plt.scatter(x,y, c='red')
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.0000001)

plt.show()
# Set useblit=True on most backends for enhanced performance.
#cursor = Cursor(ax, useblit=True, color='red', linewidth=0)

#plt.show()