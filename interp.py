from scipy.interpolate import interp2d
import numpy as np

# Known velocity values at meshgrid points
X = np.array([0, 1, 2, 3])
Y = np.array([0, 1, 2, 3])
Z = np.array([[0, 1, 2, 3], [0, 1, 4, 9], [0, 1, 8, 27], [0, 1, 14, 81]])

# Interpolation function
f = interp2d(X, Y, Z, kind='linear')

# Interpolate velocity at point (x, y) = (1.5, 2.5)
velocity = f(1.5, 2.5)

print("Velocity at (1.5, 2.5) is:", velocity)