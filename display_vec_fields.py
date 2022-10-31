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



ds = np.genfromtxt("/Users/MonicaChan/Desktop/UROP/Python Implementation/ds_vector_fields/ds3.csv", delimiter=',')
meshgrid = np.genfromtxt("/Users/MonicaChan/Desktop/UROP/Python Implementation/ds_vector_fields/meshgrid.csv", delimiter=',')
print(ds.shape)
print(meshgrid.shape)

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)
ax.add_patch(Rectangle((-100,-100), 200, 96, alpha = 0.5, color = '#ffcd4d'))
ax.add_patch(Rectangle((2,-8), 4, 14, alpha = 0.5,color = '#cc8080'))
ax.add_patch(Rectangle((2,6), 4, 2, alpha = 0.5,color="#4dcc4d"))

x = np.linspace(-8,8, num = 50)
y = np.linspace(-8,8, num = 50)
u = ds[0]
v = ds[1]

xv, yv = np.meshgrid(x,y)
u = np.reshape(u, (50,50), order = 'F')
v = np.reshape(v, (50,50), order = 'F')
ax.streamplot(xv,yv,u,v)
plt.show()