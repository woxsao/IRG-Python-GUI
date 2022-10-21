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

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)
ax.add_patch(Rectangle((-100,-100), 200, 96, alpha = 0.5, color = '#ffcd4d'))
ax.add_patch(Rectangle((2,-8), 4, 14, alpha = 0.5,color = '#cc8080'))
ax.add_patch(Rectangle((2,6), 4, 2, alpha = 0.5,color="#4dcc4d"))

#vals = np.array(list(range(0,40000)))
#vals = np.where(vals % 8 != 0)
#ds_lim = np.delete(ds, vals, axis = 1)
#meshgrid_lim = np.delete(meshgrid, vals, axis = 1)
#xv,yv = np.meshgrid(meshgrid_lim[0], meshgrid_lim[1])
#print(xv.shape)
#print(ds_lim[0].shape)
#ax.streamplot(xv,yv, ds_lim[0], ds_lim[1])
#print(ds_lim.shape)

dy = ds[1]
dx = ds[0]
x = np.linspace(-8,8, num = 40000)
y = np.linspace(-8,8, num = 40000)
print("not stuck at linspace")
xv, yv = np.meshgrid(x,y)
print("not stuck at meshgrid")
uCi = interp2d(x, y, dx)(xv, yv)
vCi = interp2d(x, y, dy)(xv, yv)
print("not stuck at interpolate")
ax.streamplot(xv,yv, uCi, vCi)
print("not stuck here")
plt.show()
#print("where?")