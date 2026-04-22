# Find inner polyhedral approximation of a nonconvex set using the maxfit algorithm. The nonconvex function is defined as a Python function, and the convex inner approximation is defined as a JAX function with learnable parameters.
#
# {x: A x <= b - err} \subseteq {x: f(x)<=0} 
#
# (C) 2026 A. Bemporad

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from jax_sysid.models import StaticModel
import jax
import jax.numpy as jnp
from maxfit import maxfit
import time

# ###############################
# MAX_ERROR_FIT parameters
# ###############################

N = 50 # initial number of samples
maxiter = 50 # maximum number of active learning iterations
warm_retrain = True  # set to True to retrain the model at each iteration starting from previous model parameters to speed up training 
gamma = 10. # soft_max function parameter
eta = 10. # soft-sign function parameter
global_optimizer = 'direct'
rho_th=1.e-4 # L2 regularization parameter
tau_th=0. # 0.01 L1 regularization parameter
model_init='mse'
plotfigs = True  # set to True to plot figures
# ###############################

np.random.seed(1) # for reproducibility of results

jax.config.update('jax_platform_name', 'cpu')
if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)  # Enable 64-bit computations

nx = 2
xmin = np.array([-2.,-2.])
xmax = -xmin
X = np.random.uniform(xmin, xmax, (N, nx))

def fun_(x):
    return x[0]**2+x[1]**4+x[0]**3/3.-1.*x[1]**3-x[0]/2.-1.
fun = np.vectorize(fun_, signature='(n)->()')

# ################################
# Polyhedron to be fitted
n1 = 10
@jax.jit
def output_fcn_(u, params):
    A, b = params
    y = jnp.max(A@u.reshape(nx,1)-b) # convex PWL function
    return y.reshape(-1)
def init_fcn(seed):
    np.random.seed(seed)
    return [np.random.randn(n1,nx), -np.random.randn(n1,1)]

output_fcn = jax.jit(jax.vmap(output_fcn_, in_axes=(0, None)))

model = StaticModel(1, nx, output_fcn)
model.optimization(adam_epochs=1000, lbfgs_epochs=1000, iprint=-1) # optimization parameters used at initialization
model.loss(rho_th=rho_th, tau_th=tau_th) # loss function used at initialization

output_fcn = jax.jit(jax.vmap(output_fcn_, in_axes=(0, None)))

# Train model
t0 = time.time()
model, results = maxfit(model, fun, X, init_fcn, lb = xmin, ub = xmax, 
                               method='set', seeds=None, maxiter=maxiter, gamma = gamma, model_init=model_init, warm_retrain=warm_retrain, adam_epochs=1000, lbfgs_epochs=1000, rho_th=rho_th, global_optimizer=global_optimizer)
t0 = time.time() - t0

MaxErr = results.MaxErr
MSErr = results.MSErr
N_act = results.N_act
X_act = results.X[-N_act:,:]
Y_act = results.Y[-N_act:]
X0 = results.X[:-N_act,:]
Y0 = results.Y[:-N_act]
worst_error = results.worst_error

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

    dx = (xmax[0] - xmin[0]) / 400.0
    dy = (xmax[1] - xmin[1]) / 400.0
    [x1, x2] = np.meshgrid(np.arange(xmin[0], xmax[0] + dx, dx), np.arange(xmin[1], xmax[1] + dy, dy))
    z = np.array([[fun_([x1[i,j], x2[i,j]]) for j in range(x1.shape[0])] for i in range(x1.shape[1])])
   
    zh = np.array([[model.predict(jnp.array([x1[i,j],x2[i,j]]).reshape(1,-1)).item() for j in range(x1.shape[0])] for i in range(x1.shape[1])])

    plt.figure(figsize=(7,7))
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.contour(x1,x2,z, levels=[0], alpha=0.7, colors=colors[0], linewidths=3)
    I_pos = np.argwhere(Y_act.reshape(-1)>0).reshape(-1)
    I_neg = np.argwhere(Y_act.reshape(-1)<=0).reshape(-1)
    plt.scatter(X_act[I_pos,0], X_act[I_pos,1], marker='$\star$', color=colors[1], s=100)
    plt.scatter(X_act[I_neg,0], X_act[I_neg,1], marker='$\star$', color=colors[2], s=100)
    I_pos = np.argwhere(Y0.reshape(-1)>0).reshape(-1)
    I_neg = np.argwhere(Y0.reshape(-1)<=0).reshape(-1)
    plt.scatter(X0[I_pos,0], X0[I_pos,1], marker='s', color=colors[1], s=50, alpha=0.7)
    plt.scatter(X0[I_neg,0], X0[I_neg,1], marker='s', color=colors[2], s=50, alpha=0.7)
    plt.contour(x1,x2,zh, levels=[0], alpha=0.7, colors=colors[3], linewidths=3)
    plt.grid()

    plt.figure(figsize=(8, 4))
    plt.plot(MaxErr, label='max error', linewidth=3)
    plt.plot(MSErr, label='mean squared error', linewidth=3)
    plt.legend()
    plt.xlabel('iteration')
    plt.grid()

print(f"Elapsed time to run training algorithm: {t0:.2f} seconds")
print(f"  training models = {results.time_training:.1f} s, global optimization: {results.time_worst_case:.1f} s")

err = -results.worst_error
print(f"err = {err:.5f}")

A,b = model.params[:-1]
b=b-err
print("The inner approximation is the polyhedron Ax <= b defined by:")
print("A = ", A)
print("b = ", b)
    
