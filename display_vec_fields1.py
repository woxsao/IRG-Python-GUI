from matplotlib.patches import Rectangle
import numpy as np
import matplotlib.pyplot as plt
import time 
from matplotlib.widgets import Button
import os
import glob
import csv
from scipy.interpolate import interp2d

ds = np.genfromtxt("/Users/MonicaChan/Desktop/UROP/Python Implementation/ds_vector_fields/ds1.csv", delimiter=',')
meshgrid = np.genfromtxt("/Users/MonicaChan/Desktop/UROP/Python Implementation/ds_vector_fields/meshgrid.csv", delimiter=',')
print(ds.shape)
print(meshgrid.shape)

fig0, ax0 = plt.subplots(figsize=(8, 8))
ax0.set_xlim(-8, 8)
ax0.set_ylim(-8, 8)
ax0.add_patch(Rectangle((-100,-100), 200, 96, alpha = 0.5, color = '#ffcd4d'))
ax0.add_patch(Rectangle((2,-8), 4, 14, alpha = 0.5,color = '#cc8080'))
ax0.add_patch(Rectangle((2,6), 4, 2, alpha = 0.5,color="#4dcc4d"))

   
# Add vector field from learned LPV-DS
#grid_size = 40
"""for i in np.linspace(-0.25, 1.25, grid_size):
    for j in np.linspace(0, 1, grid_size):
        x_query    = np.array([i, j])
        x_dot      = ds.get_ds(x_query)
        x_dot_norm = x_dot/LA.norm(x_dot) * 0.02
        plt.arrow(i, j, x_dot_norm[0], x_dot_norm[1], 
            head_width=0.008, head_length=0.01)    """


for i in range(200):
    for j in range(200):
        if (i*200+j) %1000 == 0:
            print('generating vector:', i*200+j)
        x_dot = ds[:,i*200+j]
        x_dot_norm = x_dot/(x_dot[0]**2+x_dot[1]**2)**(1/2)*0.02
        plt.arrow(meshgrid[0,i*200+j],meshgrid[1,i*200+j], x_dot_norm[0], x_dot_norm[1], head_width = 0.008,head_length = 0.01)
plt.show()