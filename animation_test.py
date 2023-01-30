import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create a 2D grid of points
X, Y = np.meshgrid(np.linspace(-5, 5, 20), np.linspace(-5, 5, 20))

# Define the vector field
U = -Y
V = X

# Plot the vector field
fig, ax = plt.subplots()
q = ax.quiver(X, Y, U, V)


# Define the function that will update the line's position at each frame
def update(num):
    # Calculate the new position of the line based on the vector field and the time elapsed
    x = line.get_xdata()[-1] + U[num]/50
    y = line.get_ydata()[-1] + V[num]/50
    line.set_data(x, y)


ani = FuncAnimation(fig, update, frames=range(len(X)), repeat=True,interval = 500)
x0, y0 = 0, 0
line, = ax.plot([x0], [y0], 'o', lw=1, color='red')


plt.show()
