import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.patches import Rectangle

def inap(x, obj, att):
    pos = obj.pos
    vertices = np.array([
        pos[:2],
        [pos[0] + pos[2], pos[1]],
        [pos[0], pos[1] + pos[3]],
        [pos[0] + pos[2], pos[1] + pos[3]]
    ])
    b = np.all(np.dot(vertices, x) <= 1) or np.linalg.norm(x - att) < 0.4
    return b

def automaton_scoop(curr_mode, x, objs, atts):
    desired_plan = [2, 3, 4, 4]
    sensor = 1
    task_succ = 0
  
    for i in range(len(objs)):
        if inap(x, objs[i], atts[i]):
            sensor = i+1
    #reaching
    if curr_mode == 1:
        if sensor == 2:
            next_mode = 2
        else:
            next_mode = 1\
    #scooping
    elif curr_mode == 2:
        if sensor == 3:
            next_mode = 3
        elif sensor == 1:
            next_mode = 1
        else:
            next_mode = 2
    #transport
    elif curr_mode == 3:
        if sensor == 1:
            next_mode = 1
        elif sensor == 2:
            next_mode = 2
        elif sensor == 3:
            next_mode = 3
        elif sensor == 4:
            next_mode = 4
            task_succ = 1
        else:
            raise ValueError(f"Wrong mode transition: {curr_mode} -> {sensor}")
    #success
    elif curr_mode == 4:
        next_mode = 4
        task_succ = 1
    else:
        raise ValueError(f"Wrong mode transition: {curr_mode} -> {sensor}")
  
    return next_mode, task_succ, desired_plan

def plot_ap():
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.add_patch(Rectangle((-100,-100), 200, 96, alpha = 0.5, color = '#ffcd4d'))
    ax.add_patch(Rectangle((2,-8), 4, 14, alpha = 0.5,color = '#cc8080'))
    ax.add_patch(Rectangle((2,6), 4, 2, alpha = 0.5,color="#4dcc4d"))
    return fig, ax
    
def start_simulation_ltl(policies, transition, opt_sim):
    start = opt_sim["start"]
    x_to_plot = np.ones(2,50) @ start
    time_step = 1
    pert_force = 0
    curr_mode = 1
    x_dot = policies[curr_mode](x)
    traj_handle = np.ndarray((2,1))
    fig, ax = plot_ap()
    while time_step > 0:
        x = x_to_plot[:,2]
        x = x + (x_dot+pert_force) * opt_sim.dt
        x_to_plot[:,1] = x
        x_query = x_to_plot[:,20]
        x_to_plot = np.roll(x_to_plot,-1)
        traj_handle = np.hstack(traj_handle,x_to_plot)
        time_step += 1
        x = traj_handle[0,:]
        y = traj_handle[1,:]
        line, = ax.plot(x, y)
        line.set_ydata(y)
        plt.draw()
        plt.pause(0.05)