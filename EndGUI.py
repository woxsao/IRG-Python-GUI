from matplotlib.patches import Rectangle
import numpy as np
import matplotlib.pyplot as plt
import time 
import os
import glob
from scipy.io import loadmat
import copy


def plot_ap():
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.add_patch(Rectangle((-100,-100), 200, 96, alpha = 0.5, color = '#ffcd4d'))
    ax.add_patch(Rectangle((2,-8), 4, 14, alpha = 0.5,color = '#cc8080'))
    ax.add_patch(Rectangle((2,6), 4, 2, alpha = 0.5,color="#4dcc4d"))
    return fig, ax


def plot_ds_model(ds, target, filenum, limits, quality='medium'):
    if quality == 'high':
        nx = 400
        ny = 400
    elif quality == 'medium':
        nx = 200
        ny = 200
    else:
        nx = 50
        ny = 50
    fig,ax = plot_ap()
    axlim = limits
    ax_x = np.linspace(axlim[0], axlim[1], nx)
    ax_y = np.linspace(axlim[2], axlim[3], ny)
    x_tmp, y_tmp = np.meshgrid(ax_x, ax_y)
    x = np.vstack((y_tmp.flatten(),x_tmp.flatten()))
    x_ = x
    xd = ds(x_)
    np.savetxt("lpv_ds"+str(filenum)+".csv", xd, delimiter=',')
    #TODO: Compare the lpv_ds function outputs here:
    #file = os.getcwd()+"/dsltl/experiments/scoop_seed_04/lpv_ds"+str(filenum)+".csv"
    #matlpvds = np.loadtxt(file, delimiter=',')
    #print("matlpvds shape:", matlpvds.shape)
    #print("xd shape:", xd.shape)
    #matlpvds_sorted = np.sort(matlpvds, axis = 1)
    #xd_sorted = np.sort(xd,axis = 1)
    #print("Are arrays equal?", np.isclose(matlpvds_sorted, xd_sorted, rtol = 1e-3, atol = 1e-3))


    h = plt.streamplot(
        x_tmp, y_tmp, xd[0, :].reshape((ny, nx)), xd[1, :].reshape((ny, nx)),
        density=1,
        color='gray',
        linewidth=0.75
    )
    h_att = plt.plot(target[0], target[1], 'o')
    h_att[0].set_markerfacecolor([1, 0.5, 0])
    h_att[0].set_markersize(15)
    h_att[0].set_markeredgecolor([1, 0.5, 0])
    return h, xd

def lpv_ds(x, ds_gmm, A_g, b_g,filenum):
    #print("x shape: ", filenum)
    N, M = np.shape(x)
    K = len(ds_gmm['Priors'][0][0][0])
    x_ = np.array([x[1,:],x[0,:]])
    beta_k_x = posterior_probs_gmm(x_, ds_gmm, 'norm')
    x_dot = np.zeros((N, M))

    for i in range(M):
        if b_g.shape[1] > 1:
            f_g = np.zeros((N, K))
            for k in range(K):
                x_p = [x[1,i], x[0,i]]
                f_g[:,k] = np.multiply(beta_k_x[k,i], (np.matmul(A_g[:,:,k], x_p)) + b_g[:,k])
            f_g = np.sum(f_g, axis=1)
        else:
            f_g = A_g @ x[:, i] + b_g
        x_dot[:, i] = f_g


    return x_dot

def posterior_probs_gmm(x, gmm, type):
    N, M = np.shape(x)
    Mu = np.array(gmm['Mu'])[0][0]
    Priors = gmm['Priors'][0][0][0]
    Sigma = gmm['Sigma'][0][0]
    #print("priros: ", Priors.shape)
    #print(len(Priors))
    K = len(Priors)
    # Compute mixing weights for multiple dynamics
    Px_k = np.zeros((K, M))
    # Compute probabilities p(x^i|k)
    for k in range(K):
        #Px_k[k,:] = np.apply_along_axis(lambda x_i: multivariate_normal.pdf(x_i, mean=Mu[:,k], cov=Sigma[:,:,k]), axis=0, arr=x)+eps
        #Px_k[k,:] = np.apply_along_axis(lambda x_i: multivariate_normal.pdf(x_i, mean=Mu[:,k], cov=Sigma[:,:,k]), axis=0, arr=x) + np.finfo(np.longdouble).eps
        Px_k[k,:] = ml_gaussPDF(x,Mu[:,k],Sigma[:,:,k])+np.spacing(1)
    # Compute posterior probabilities p(k|x)
    alpha_Px_k = np.tile(Priors.T, (M,1)).T * Px_k
    if type == 'norm':
        #Pk_x = alpha_Px_k / np.sum(alpha_Px_k, axis=0, keepdims=True)
        sums = np.sum(alpha_Px_k, axis=0, keepdims=True)
        sums_repeated = np.tile(sums, (K,1))
        Pk_x = alpha_Px_k / sums_repeated
    elif type == 'un-norm':
        Pk_x = alpha_Px_k
    else:
        raise ValueError("Invalid type. Must be 'norm' or 'un-norm'.")

    return Pk_x


def ml_gaussPDF(Data, Mu, Sigma):
    # (D x N) - repmat((D x 1),1,N)
    # (D x N) - (D x N)
    nbVar, nbData = Data.shape
    Mu = np.reshape(Mu,(Mu.shape[0],1))
    Mus = np.tile(Mu, (1, nbData))
    Data = (Data - Mus).T
    # Data = (N x D)
    
    # (N x 1)
    prob = np.sum((Data @ np.linalg.inv(Sigma)) * Data, axis=1)
    prob = np.exp(-0.5 * prob) / np.sqrt((2 * np.pi) ** nbVar * abs(np.linalg.det(Sigma) + np.finfo(float).tiny)) + np.finfo(float).tiny
    return prob

def plot_policy(pol, limits):
    nx = 50
    ny = 50
    axlim = limits
    ax_x = np.linspace(axlim[0], axlim[1], nx)
    ax_y = np.linspace(axlim[2], axlim[3], ny)
    x_tmp, y_tmp = np.meshgrid(ax_x,ax_y)
    x = np.vstack((y_tmp.flatten(),x_tmp.flatten()))
    xd = pol(x)
    print('policy in plot pol', pol)
    h = plt.streamplot(
        x_tmp, y_tmp, xd[0, :].reshape((ny, nx)), xd[1, :].reshape((ny, nx)),
        density=1,
        color='gray',
        linewidth=0.75
    )
    return h,x

def plot_multi_mode(hs, pol, limits, att, visited, failure, cut_normals):
    for j in range(len(hs)):
        del(hs[i])
    #plot vec field
    policy_handle,vec_field = plot_policy(pol,limits)
    #plot att
    att_handle = plt.plot(att[0], att[1], 'o')
    att_handle[0].set_markerfacecolor([1, 0.5, 0])
    att_handle[0].set_markersize(15)
    att_handle[0].set_markeredgecolor([1, 0.5, 0])

    return policy_handle,vec_field
    #TODO: Plot cuts

def vector_field(x,y,pol):
    global ds_debug
    pt = np.reshape(np.stack((y,x)),(2,1))
    print('policy', pol)
    return ds_debug[0](pt)

def start_simulation_ltl(policies, transition, opt_sim):
    global X,Y
    curr_mode = 1
    atts = opt_sim["atts"]
    start = opt_sim["start"]
    fig, ax = plot_ap()
    hs = []
    #curr_mode, task_succ, target_mode= transition(curr_mode, start, objs, atts)
    objs.append(objs[-1])
    hs,vec_field = plot_multi_mode(hs,policies[curr_mode-1],[-8,8,-8,8],atts[curr_mode-1],[],[],[])
    time_step = 1
    start = np.reshape(start,(2,1))
    x_to_plot = start
    mod_policies = []
    for k in range(4):
        mod_policies.append(policies[k])
    line, = ax.plot(X, Y, 'g', lw=2)
    
    def update(frame,x_to_plot,curr_mode):
        global X,Y
        pt = np.reshape(np.stack((x_to_plot[1,0],x_to_plot[0,0])),(2,1))
        vel = policies[curr_mode-1](pt)
        dx, dy = vel[0,:], vel[1,:]
        
        x_ = x_to_plot[0,0]
        y = x_to_plot[1,0]
        x_ += dx * 0.01
        y += dy * 0.01
        x_to_plot = np.vstack((x_,y))
        X.append(x_)
        Y.append(y)
        line.set_data(X[:frame], Y[:frame])
        return x_to_plot
    start_time = time.time() 
    curr_mode, task_succ, target_mode = transition(curr_mode, start, objs, atts)
    while time_step >0:
        #print('curr mode', curr_mode)
        x_to_plot = update(time_step,x_to_plot,curr_mode)
        if not task_succ:
            next_mode, task_succ, desired_plan = transition(curr_mode, x_to_plot, objs, atts)
            #print("next mode:", next_mode)
            if not task_succ and next_mode != curr_mode:
                fig,ax = plot_ap()
                X = X[-1:]
                Y = Y[-1:]
                line, = ax.plot(X, Y, 'g', lw=2)
                hs = plot_multi_mode([], mod_policies[next_mode-1], limits, atts[next_mode-1],"visited","failure","cut normals")
                #print("MDOE TRANSITIONONONONON")
                curr_mode = next_mode
        else:
            time_step = -100
        plt.draw()
        plt.pause(0.01)
        time_step +=1
    print("out!")
    return fig,ax, vec_field,policies[curr_mode-1]
def inap(x, obj, att):
    pos = obj["pos"]
    vertices = np.array([
        pos[:2], #top left
        [pos[0] + pos[2], pos[1]], #bottom left
        [pos[0], pos[1] + pos[3]], #top right
        [pos[0] + pos[2], pos[1] + pos[3]] #bottom right
    ])
    b = is_point_inside_rectangle(x, vertices)
    norm_diff = np.linalg.norm(x - att)
    #print("vertices:", vertices, "x: ",x)
    #print("b",b)
    return b or norm_diff < 0.4

def is_point_inside_rectangle(point, vertices):
    x, y = point[0,0], point[1,0]
    x_min, x_max = min(vertices[:, 0]), max(vertices[:, 0])
    y_min, y_max = min(vertices[:, 1]), max(vertices[:, 1])
    return x_min <= x <= x_max and y_min <= y <= y_max


def automaton_scoop(curr_mode, x, objs, atts):
    desired_plan = [2, 3, 4, 4]
    sensor = 1
    task_succ = 0
    for i in range(1,len(objs)):
        if inap(x, objs[i-1], atts[i-1]):
            sensor = i+1
    #reaching
    if curr_mode == 1:
        if sensor == 2:
            next_mode = 2
        else:
            next_mode = 1
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




"""====================================MAIN CODE========================================================================"""
"""====================================================================================================================="""
"""====================================================================================================================="""
"""====================================================================================================================="""
"""====================================================================================================================="""
objs = [{"symbol":"R", "pos":[-100,-100,200,96]},{"symbol":"S", "pos":[2,-8,4,14]},{"symbol":"T", "pos":[2,6,4,2]}]
models = []
limits = [-8,8,-8,8]
directory = os.getcwd()+"/dsltl/experiments/scoop_seed_04/"
files = glob.glob(directory + "/ds0*")
ds_debug = []
#Load DS
i = 0
ds_list = []
for f in sorted(files):
    def closure(b):
        return lambda x: lpv_ds(x, copy.deepcopy(ds["ds_gmm"]), copy.deepcopy(ds["A_k"]), copy.deepcopy(ds["b_k"]),b)
    filename = directory + os.path.basename(f)
    print(filename)
    ds = loadmat(filename)
    ds_list.append(ds)
    ds["ds_lpv"] = closure(i)
    models.append(ds.copy())
    #hs = plot_ds_model(ds["ds_lpv"], ds["att"], i, limits,'low')
    
    i+=1

print("models", models[0]["b_k"],models[1]["b_k"],models[2]["b_k"])

#TODO: If I cahnge b to i it works, for some reason b doesn't work, weird pass by reference issue potentially
ds_debug = []
for i in range(3):
    ds_debug.append(lambda x,i=i: lpv_ds(x,models[i]["ds_gmm"],models[i]["A_k"], models[i]["b_k"],i))
    print("identity:", ds_debug[i])
ds_debug.append(ds_debug[-1])
segs = []
files = glob.glob(directory + "/seg0*")
for f in sorted(files):
    seg = {}
    filename = directory + os.path.basename(f)
    print(filename)
    s = loadmat(filename)["seg"][0][0]
    seg["drawn_data"] = s[0]
    seg["Data"] = s[1]
    seg["Data_sh"] = s[2]
    seg['att'] = s[3]
    seg['x0_all'] = s[4]
    seg['dt'] = s[5][0,0]
    segs.append(seg)


opt_sim = {}
opt_sim["dt"] = 0.005
opt_sim["i_max"] = 10000
opt_sim["tol"] = 0.001
opt_sim["plot"] = 0
opt_sim["start"] = segs[0]["x0_all"][:,0]

atts = []
for i in range(3):
    atts.append(segs[i]["att"])
opt_sim["atts"] = atts
opt_sim["pert_tag"] = 'vel'
opt_sim["if_mod"] = 1
opt_sim["if_plot"] = 1
opt_sim["add_noise"] = 0

transition = lambda mode,x,objs,atts: automaton_scoop(mode,x,objs,atts)

T = 50
dt = 0.01
x0 = opt_sim["start"][0]
y0 = opt_sim["start"][1]
x = x0
y = y0
X = [x0]
Y = [y0]
fig,ax,_, new_pol = start_simulation_ltl(ds_debug,transition, opt_sim)
