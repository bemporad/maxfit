'''
MaxFit - Example: approximate a nonlinear 2D function.

(C) 2026 A. Bemporad
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from jax_sysid.models import StaticModel
import jax
import jax.numpy as jnp
from flax import linen as nn
from maxfit import maxfit, uncertainty_bounds
import time
from functools import partial

np.random.seed(1) # for reproducibility of results

jax.config.update('jax_platform_name', 'cpu')
if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)  # Enable 64-bit computations

plotfigs = True

# ########################################
# MaxFit parameters
# ########################################

N = 100 # initial number of samples
maxiter=50 # maximum number of iterations (number of corner-case samples to add)
max_error = 1.e-6 # minimum maximum error achieved over the set of x to stop the optimization
gamma = 10. # soft_max function parameter
nu = 0.e-2 # weight on MSE term added on top of the soft-maximum loss
max_steps_no_improvement=maxiter # stop if there's no improvement for this number of steps

# ########################################
# Test function and data generation
# ########################################
nx = 2
xmin = 0.
xmax = 1.
ymin = 0.
ymax = 1.
X = np.random.rand(N, nx) * np.array([xmax - xmin, ymax - ymin]) + np.array([xmin, ymin])

def fun_2d(x1, x2):
    return np.exp(-30.*((x1-0.5)**2+(x2-0.5)**2))
def fun(x):
    return fun_2d(x.reshape(-1)[0],x.reshape(-1)[1])

# ########################################
# Model parameters
# ########################################
n1 = 10 # NN layer 1
n2 = 5 # NN layer 2
act = partial(nn.leaky_relu, negative_slope=0.01)
rho_th=1.e-8
tau_th=0.
warm_retrain=False
model_init='mse' # model initialization method

@jax.jit
def output_fcn_(u, params):
    V1, b1, W2, V2, b2, W3, V3, b3 = params
    y = V1@u.reshape(-1,1)+b1
    y = W2@act(y)+V2@u.reshape(-1,1)+b2
    y = W3@act(y)+V3@u.reshape(-1,1)+b3
    y = jnp.maximum(y, 0.)
    return y.reshape(-1)
output_fcn = jax.jit(jax.vmap(output_fcn_, in_axes=(0, None)))

model = StaticModel(1, nx, output_fcn)
model.optimization(adam_epochs=1000, lbfgs_epochs=1000, iprint=-1) # optimization parameters used at initialization
model.loss(rho_th=rho_th, tau_th=tau_th) # loss function used at initialization

# ########################################
# Error envelope parameters
# ########################################
uncertainty = 'variable-asymmetric'
uncertainty_neurons = [20,10]  # neurons to use for uncertainty estimation
uncertainty_activation = act  # activation function for uncertainty estimation
rho_psi = 1.e-8
global_optimizer = 'direct'
N_act_repeat=10

def init_fcn(seed):
    np.random.seed(seed)
    return [np.random.randn(n1, nx), np.random.randn(
    n1, 1), np.random.randn(n2, n1), np.random.randn(n2, nx), np.random.randn(n2, 1), np.random.randn(1, n2), np.random.randn(1, nx), np.random.rand(1, 1)]

lb=np.array([xmin,ymin])
ub=np.array([xmax,ymax])

# ###################################
# Train model via worst-case fitting
# ###################################
t0 = time.time()
model, results = maxfit(model, fun, X, init_fcn, lb = lb, ub = ub, method='function', seeds=None, maxiter=maxiter, max_error=max_error, gamma = gamma, nu=nu, model_init=model_init, adam_epochs=100, lbfgs_epochs=100, rho_th=rho_th, warm_retrain=warm_retrain, global_optimizer=global_optimizer, N_act_repeat=N_act_repeat)
t0 = time.time() - t0

# Find uncertainty bounds
print(f"Fitting uncertainty bounds with method {uncertainty} ... ", end='')
# Get a new set of points to estimate the uncertainty bounds
N1 = 10000
X1 = np.random.rand(N1, nx) * np.array([xmax - xmin, ymax - ymin]) + np.array([xmin, ymin])
Y1 = np.array([fun(x) for x in X1])

X_epsil = results.X 
Y_epsil = results.Y.reshape(-1,1)
worst_error = results.worst_error
t1 = time.time()
model_epsilon_u, model_epsilon_ell = uncertainty_bounds(model, fun, X_epsil, Y_epsil, lb, ub, worst_error, uncertainty=uncertainty, uncertainty_neurons=uncertainty_neurons, uncertainty_activation=uncertainty_activation, rho=rho_psi, alpha=1.e-3, gamma_eps=100., global_optimizer=global_optimizer, adam_epochs=1000, lbfgs_epochs=1000)
t1 = time.time() - t1

MaxErr = results.MaxErr
MSErr = results.MSErr
N_act = results.N_act
X_act = results.X[-N_act:,:]
Y_act = results.Y[-N_act:]
X0 = results.X[:-N_act,:]
Y0 = results.Y[:-N_act]
    
print(f"Elapsed time: total = {t0+t1:.1f} s, Algorithm~1 = {t0:.1f} s, uncertainty bounds = {t1:.1f} s")
print(f"              training models = {results.time_training:.1f} s, global optimization: {results.time_worst_case:.1f} s")


if plotfigs:
    plt.close('all')    
    mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 20,
    "font.size": 20,
    "legend.fontsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    })
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    dx = (xmax - xmin) / 100.0
    dy = (ymax - ymin) / 100.0
    [x1, x2] = np.meshgrid(np.arange(xmin, xmax + dx, dx), np.arange(ymin, ymax + dy, dy))
    z = np.array([[fun(np.array([x1[i,j], x2[i,j]])) for j in range(x1.shape[1])] for i in range(x1.shape[0])])

    fig, ax = plt.subplots(figsize=(8, 8))
    nlevels=20
    plt.contour(x1, x2, z, alpha=0.4, levels=nlevels, cmap ='terrain')
    plt.contourf(x1, x2, z, levels=nlevels, alpha=0.4, cmap ='terrain')
    plt.title(r'level sets of $f(x)$', fontsize=20)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    plt.scatter(X_act[:,0],X_act[:,1], s=150, marker='$\star$', alpha=1., color=colors[1], label='corner cases', linewidths=1.5)
    ax.scatter(X0[:,0],X0[:,1],label='initial samples', color='gray', alpha=1.0, linewidths=1.5)
    plt.grid()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(r'$x_1$', labelpad=15, fontsize=24)
    ax.set_ylabel(r'$x_2$', labelpad=15, fontsize=24)
    ax.set_zlabel(r'$y$', fontsize=24, labelpad=20)
    ax.plot_surface(x1, x2, z, alpha=0.5, cmap = 'managua')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    plt.suptitle('$f(x)$', fontsize=24, y=0.87)

    ticks = ax.get_yticks()
    labels = [str(int(t*10)/10) if i>0 else '' for i, t in enumerate(ticks)]
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)

    fig, ax = plt.subplots(2,1, figsize=(8,5))
    ax[0].plot(np.arange(N,N+maxiter),MaxErr, label="WCE", linewidth=3, color=colors[0])
    ax[1].plot(np.arange(N,N+maxiter),MSErr, label="MSE", linewidth=3, color=colors[1])
    for i in range(2):
        if i==1:
            ax[i].set_xlabel(f'iteration $N$')
        ax[i].legend()
        ax[i].grid()
    ax[0].tick_params(labelbottom=False)

    dx = (xmax - xmin) / 100.0
    dy = (ymax - ymin) / 100.0
    [x1, x2] = np.meshgrid(np.arange(xmin, xmax + dx, dx), np.arange(ymin, ymax + dy, dy))
    zhat = model.predict(np.hstack((x1.reshape(-1,1), x2.reshape(-1,1)))).reshape(x1.shape)

    fig, ax = plt.subplots(figsize=(8, 8))
    nlevels=30
    plt.contour(x1, x2, zhat, alpha=0.4, levels=nlevels, cmap ='terrain')
    plt.contourf(x1, x2, zhat, levels=nlevels, alpha=0.4, cmap ='terrain')
    plt.title(r'level sets of $\hat f(x)$', fontsize=20)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    plt.scatter(X_act[:,0],X_act[:,1], s=150, marker='$\star$', alpha=1., color=colors[1], label='corner cases', linewidths=1.5)
    ax.scatter(X0[:,0],X0[:,1],label='initial samples', color='gray', alpha=1.0, linewidths=1.5)
    plt.grid()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(r'$x_1$', labelpad=15, fontsize=24)
    ax.set_ylabel(r'$x_2$', labelpad=15, fontsize=24)
    ax.set_zlabel(r'$y$', fontsize=24, labelpad=20)
    ax.plot_surface(x1, x2, zhat, alpha=0.5, cmap = 'managua')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    plt.suptitle('$\hat f(x)$', fontsize=24, y=0.87)

    ticks = ax.get_yticks()
    labels = [str(int(t*10)/10) if i>0 else '' for i, t in enumerate(ticks)]
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    dx = (xmax - xmin) / 50.0
    dy = (ymax - ymin) / 50.0
    [x1, x2] = np.meshgrid(np.arange(xmin, xmax + dx, dx), np.arange(ymin, ymax + dy, dy))
    xx = np.hstack((x1.reshape(-1,1), x2.reshape(-1,1)))
    eps_hat_u = model_epsilon_u.predict(xx).reshape(x1.shape)
    eps_hat_ell = model_epsilon_ell.predict(xx).reshape(x1.shape)
    surf1 = ax.plot_surface(x1, x2, eps_hat_u, cmap='Blues', alpha=0.5, label=r'$\varepsilon^u(x)$')
    surf2 = ax.plot_surface(x1, x2, -eps_hat_ell, cmap='Reds', alpha=0.5, label=r'$\varepsilon^\ell(x)$')

    E = np.array([fun(x) for x in xx]).reshape(x1.shape)-model.predict(xx).reshape(x1.shape)
    surf3 = ax.plot_surface(x1, x2, E, cmap='Greens', alpha=0.4, label=r'$\varepsilon^\ell(x)$')
    
    E0 = Y0-model.predict(X0).reshape(-1)
    ax.scatter(X0[:,0],X0[:,1],E0, label='initial samples', color='gray', alpha=1.0, depthshade=False, linewidths=1.5)

    E_act = np.array([fun(x) for x in X_act]).reshape(-1)-model.predict(X_act).reshape(-1)
    ax.scatter(X_act[:,0],X_act[:,1],E_act, s=150, marker='$\star$', alpha=1., color=colors[1], label='corner cases', depthshade=False, linewidths=1.5)
    fig.suptitle('uncertainty interval', fontsize=24, y=0.8)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.view_init(8., -140.)
    
    ticks = ax.get_yticks()
    labels = [str(int(t*10)/10) if i>0 else '' for i, t in enumerate(ticks)]
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.tick_params(axis='z', pad=8)
    