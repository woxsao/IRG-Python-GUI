import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define a simple vector field
def vector_field(x, y):
    return -y, x

# Evaluate the vector field at evenly spaced points in a grid
dx = 0.25
dy = 0.25
x = np.arange(-3, 3, dx)
y = np.arange(-3, 3, dy)
X, Y = np.meshgrid(x, y)
U, V = vector_field(X, Y)

# Create a figure and axis for the plot
fig, ax = plt.subplots()

# Plot the vector field
quiver = ax.quiver(X, Y, U, V, color='r')

# Integrate the differential equations to get the trajectory
T = 50
dt = 0.01
x0 = 0.5
y0 = 1
x = x0
y = y0
X = [x0]
Y = [y0]
line, = ax.plot(X, Y, 'g', lw=2)

# Define the animation update function
def update(frame):
    global x, y, X, Y
    dx, dy = vector_field(x, y)
    x += dx * dt
    y += dy * dt
    X.append(x)
    Y.append(y)
    line.set_data(X[:frame], Y[:frame])
    return line,

# Create the animation
ani = FuncAnimation(fig, update, frames=int(T/dt), blit=True)

# Show the animation
plt.show()
