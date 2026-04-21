'''
MaxFit - Example: approximate a scalar function.

(C) 2026 A. Bemporad
'''

import numpy as np
import matplotlib.pyplot as plt
from jax_sysid.models import StaticModel
import jax
from flax import linen as nn
import time
from maxfit import maxfit, uncertainty_bounds

np.random.seed(0) # for reproducibility of results

jax.config.update('jax_platform_name', 'cpu')
if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)  # Enable 64-bit computations


plotfigs = True  # set to True to plot figures

def fun_1d(x):
    return (np.sin(x*(1.-x/10.))+(x/10.)**3-4.*x/10.)*np.exp(-x)/(1.+np.exp(-x))

# ########################################
# MaxFit parameters
# ########################################
N = 20 # initial number of samples
maxiter=30
warm_retrain=False
model_init='mse'
gamma = 10. # soft_max function parameter
nu = 0. # weight on MSE term added on top of the soft-maximum loss
global_optimizer = 'direct'
rho_th=1.e-8
tau_th=0.

# ########################################
# Test function and data generation
# ########################################
nx = 1
xmin, xmax = -10., 10.

X = np.random.uniform(xmin, xmax, N).reshape(-1, 1)

def fun(x):
    return fun_1d(x[0])

# ########################################
# Model parameters
# ########################################
n1=2 # NN layer 1
n2=1 # NN layer 2
act = nn.tanh 
uncertainty_neurons = [2,1]  # neurons to use for uncertainty estimation
uncertainty_activation = nn.tanh  # activation function for uncertainty estimation
rho_epsilon = 1.e-6  # L2 regularization on the parameters of the uncertainty bound function(s)

@jax.jit
def output_fcn_(u, params):
    V1, b1, W2, V2, b2, W3, V3, b3 = params
    y = V1@u.reshape(-1,1)+b1
    y = W2@act(y)+V2@u.reshape(-1,1)+b2
    y = W3@act(y)+V3@u.reshape(-1,1)+b3
    return y.reshape(-1)
output_fcn = jax.jit(jax.vmap(output_fcn_, in_axes=(0, None)))

model = StaticModel(1, nx, output_fcn)
model.optimization(adam_epochs=1000, lbfgs_epochs=1000, iprint=-1)
model.loss(rho_th=rho_th, tau_th=tau_th)

def init_fcn(seed):
    np.random.seed(seed)
    return [np.random.randn(n1, nx), np.random.randn(
    n1, 1), np.random.randn(n2, n1), np.random.randn(n2, nx), np.random.randn(n2, 1), np.random.randn(1, n2), np.random.randn(1, nx), np.random.rand(1, 1)]
lb=np.array([xmin])
ub=np.array([xmax])

# Train model
t0 = time.time()
model, results = maxfit(model, fun, X, init_fcn, lb = lb, ub = ub, method='function',seeds=None, maxiter=maxiter, max_error=None, gamma = gamma, nu=nu, model_init=model_init, warm_retrain=warm_retrain, adam_epochs=1000, lbfgs_epochs=1000, rho_th=rho_th, global_optimizer=global_optimizer)
t0 = time.time() - t0

MaxErr = results.MaxErr
MSErr = results.MSErr
N_act = results.N_act
X_act = results.X[-N_act:]
Y_act = results.Y[-N_act:]
worst_error = results.worst_error

if plotfigs:
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10

    fig,ax = plt.subplots(1,2, figsize=(8, 2))
    ax[0].plot(N+np.arange(maxiter),MaxErr)
    ax[0].set_xlabel('iteration $N$')
    ax[0].set_title('maximum error $e_N$')    
    ymin, ymax = ax[0].get_ylim()
    ax[0].fill_between(np.arange(N+1), np.ones(N+1)*ymin, np.ones(N+1)*ymax, color='gray', alpha=0.2)
    ax[0].grid()

    ax[1].set_title("acquired samples")
    ax[1].set_xlabel(r'iteration $N$')
    ax[1].scatter(N+np.arange(maxiter), X_act.reshape(-1), color='orange')
        
    ymin, ymax = ax[1].get_ylim()
    ax[1].scatter(np.arange(N), X.reshape(-1), color='gray', alpha=0.5)
    ax[1].fill_between(np.arange(N+1), np.ones(N+1)*ymin, np.ones(N+1)*ymax, color='gray', alpha=0.2)
    ax[1].grid()
        
x = np.arange(xmin, xmax+(xmax-xmin)/100, (xmax-xmin)/100).reshape(-1, 1)
y = np.array([fun(xi) for xi in x]).reshape(-1, 1)
yhat = model.predict(x)

# ########################################
# Error envelopes
# ########################################
uncertainty = 'variable-asymmetric' # different input-dependent upper and lower bound functions on the prediction error
i1=0

X_epsil = np.vstack((results.X,x)) # use all available data for uncertainty estimation
Y_epsil = np.vstack((results.Y.reshape(-1,1),y)) # use all available data for uncertainty estimation

print(f"Fitting uncertainty bounds with method {uncertainty} ... ", end='')
model_eps_u, model_eps_ell = uncertainty_bounds(model, fun, X_epsil, Y_epsil, lb, ub, worst_error, uncertainty=uncertainty, uncertainty_neurons=uncertainty_neurons, uncertainty_activation=uncertainty_activation, rho=rho_epsilon, alpha=1.e-3, gamma_eps=100., global_optimizer=global_optimizer, adam_epochs=2000, lbfgs_epochs=2000)

if plotfigs:
    eps_u = model_eps_u.predict(x.reshape(-1,1)).reshape(-1)
    eps_ell = model_eps_ell.predict(x.reshape(-1,1)).reshape(-1)
    E = np.array([fun(xi)-model.predict(xi.reshape(1,-1)).item() for xi in X])
    E_act = np.array([fun(xi)-model.predict(xi.reshape(1,-1)).item() for xi in X_act])

    fig, ax = plt.subplots(2,1,figsize=(5,7))
    ax[0].plot(x, y, label=r'$f(x)$')
    ax[0].plot(x, yhat, label=r'$\hat f(x)$')
    ax[0].plot(x, yhat.reshape(-1)+eps_u, color='red', linewidth=0.1)
    ax[0].plot(x, yhat.reshape(-1)-eps_ell, color='red', linewidth=0.1)
    ax[0].fill_between(x.reshape(-1), yhat.reshape(-1)+eps_u, yhat.reshape(-1)-eps_ell, color='red', alpha=0.2)
    ax[0].legend(fontsize=12)
    ax[0].grid()
    ax[0].set_title('worst-case fit', fontsize=14)

    ax[1].plot(x, y-yhat.reshape(-1,1))
    ax[1].set_title(r'$f(x)-\hat f(x)$')
    ax[1].set_xlabel(r'$x$', fontsize=16)
    ax[1].scatter(X.reshape(-1), E.reshape(-1), marker='.', color='black', alpha=0.5, s=50, label='initial samples')
    ax[1].scatter(X_act.reshape(-1), E_act.reshape(-1), marker='*', color='red', s=50, label='active samples')
    ax[1].grid()
    ax[1].plot(x, eps_u, color='red', linewidth=0.1)
    ax[1].plot(x, -eps_ell, color='red', linewidth=0.1)
    ax[1].fill_between(x.reshape(-1), eps_u, -eps_ell, color='red', alpha=0.2)
    ax[1].legend(fontsize=12)
    print("done.")
