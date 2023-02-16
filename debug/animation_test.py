import numpy as np


def is_point_inside_rectangle(point, vertices):
    x, y = point[0,0], point[1,0]
    x_min, x_max = min(vertices[:, 0]), max(vertices[:, 0])
    y_min, y_max = min(vertices[:, 1]), max(vertices[:, 1])
    return x_min <= x <= x_max and y_min <= y <= y_max

vertices = np.array([[-100,-100],
 [ 100,-100],
 [-100,-4],
 [ 100,-4]])

point = np.array([[-3.05395928],
 [-4.26455826]])

print(is_point_inside_rectangle(point, vertices))