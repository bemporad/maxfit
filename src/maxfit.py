'''
MaxFit - Solve worst-case nonlinear regression problems by approximating a given function using active learning of corner-case samples via global optimization to minimize the worst-case approximation error.

The approach also allows finding an inner approximation of the set {x: fun(x)<=0} with a surrogate {x: fhat(x)<=0}.

The algorithm works as follows:

1. Fit a nonlinear model to data using a soft maximum-error as loss function.

2. Use global optimization to find a new datapoint that maximizes the maximum error.

3. Retrain the model on the extended dataset.

4. Repeat steps 2-3 until the maximum error is below a threshold or there's no improvement.

After fitting the model, uncertainty bounds on the prediction error can be computed, either as a constant bound, or by training a second model to approximate the input-dependent error bound function, scaled by a scaling factor determined by global optimization to ensure that the bound covers the prediction error on the entire input set.

(C) 2026 A. Bemporad
'''

import numpy as np
from jax_sysid.models import find_best_model, StaticModel
import jax
import flax.linen as nn
import jax.numpy as jnp
from joblib import Parallel, delayed, cpu_count
import nlopt
from copy import deepcopy
from types import SimpleNamespace
import time

_NLOPT_ALGOS = {
    'direct':      nlopt.GN_DIRECT,
    'direct-l':    nlopt.GN_DIRECT_L,
    'orig-direct': nlopt.GN_ORIG_DIRECT,
    'crs':         nlopt.GN_CRS2_LM,
    'mlsl':        nlopt.GN_MLSL,
    'ags':         nlopt.GN_AGS,
    'isres':       nlopt.GN_ISRES,
    'esch':        nlopt.GN_ESCH,
}


def solve_global_optimization(globopt_loss, lb, ub, global_optimizer = 'direct', ftol_abs = 1e-8, maxeval = 2000, xtol_rel = 1e-5):
    """Solves the global optimization problem min_x globopt_loss(x) subject to lb <= x <= ub, using the specified global optimization method.

    The optimization is parallelized by splitting the search space into N_parallel rectangles and running the global optimizer on each rectangle in parallel, where N_parallel is the number of available CPU cores. The best solution among the parallel runs is returned. 
    
    Parameters:
    -----------
    globopt_loss: function
        The loss function to be minimized, which takes as input a vector x and returns a scalar value.
    lb: array-like
        The lower bounds for the optimization variables.
    ub: array-like
        The upper bounds for the optimization variables.
    global_optimizer: str
        The global optimization method to be used. Must be one of the keys in the _NLOPT_ALGOS dictionary.
    ftol_abs: float
        The absolute tolerance on the function value.
    maxeval: int
        The maximum number of evaluations.
    xtol_rel: float
        The relative tolerance on the optimization variables.

    Returns:
    -----------
    x_new: array
        The optimal solution found by the global optimization process.
    f_opt: float
        The optimal value of the loss function at x_new.
          
    (C) 2026 A. Bemporad
    """
    if global_optimizer not in _NLOPT_ALGOS:
        raise ValueError(f'Unknown global optimizer "{global_optimizer}"')
    N_parallel = cpu_count()

    def solve(k):
        nx = lb.size
        lb0 = lb[0]+(ub[0]-lb[0])*k/N_parallel
        ub0 = lb[0]+(ub[0]-lb[0])*(k+1)/N_parallel
        lbx = np.hstack((lb0, lb[1:]))
        ubx = np.hstack((ub0, ub[1:]))

        opt = nlopt.opt(_NLOPT_ALGOS[global_optimizer], nx)
        opt.set_lower_bounds(lbx.reshape(-1))
        opt.set_upper_bounds(ubx.reshape(-1))
        opt.set_ftol_abs(ftol_abs)
        opt.set_maxeval(maxeval)
        opt.set_xtol_rel(xtol_rel)
        opt.set_min_objective(lambda x, _: globopt_loss(x))
        x_opt = opt.optimize(((lbx+ubx)/2.).reshape(-1))
        f_val = globopt_loss(x_opt)
        del opt
        return {"x": x_opt, "f": f_val}

    results = Parallel(n_jobs=N_parallel)(delayed(solve)(k) for k in range(N_parallel))
    best = np.argmin([results[i]["f"] for i in range(N_parallel)])
    x_new = results[best]["x"]
    f_opt = results[best]["f"]
    return x_new, f_opt

def _scale_last_layer(model, factor):
    model.params[-2] *= factor
    model.params[-1] *= factor

def uncertainty_bounds(model, fun, X, Y, lb=None, ub=None, worst_error=1.e8, uncertainty='constant-symmetric',
                       uncertainty_neurons=[5,3], uncertainty_activation=nn.swish, alpha=1.e-3, gamma_eps=100., rho=1.e-6, seeds=None, adam_epochs=1000, lbfgs_epochs=1000, iprint=-1, global_optimizer='direct'):
    """
    Computes uncertainty bounds on the prediction error of a model using active learning of corner-case samples via global optimization.

    Parameters:
    -----------
    model: StaticModel
        Model to be trained. This is a StaticModel object from the jax_sysid library.
    fun: function
        Function to be approximated. This must have the form y=fun(x) where x is a vector of inputs and y is a scalar or vector output.
    X: array
        Set of feature vectors.
    Y: array
        Set of output values corresponding to X.
    lb: array or None
        Lower bounds for the input variables.
    ub: array or None
        Upper bounds for the input variables.
    worst_error: float
        Worst-case error found during active learning.
    uncertainty: string
        'constant-symmetric' = uniform uncertainty bound on absolute prediction error
        'constant-asymmetric' = uniform lower and upper bounds on the prediction error
        'variable-symmetric'  = input-dependent bound function on absolute prediction error
        'variable-asymmetric' = different input-dependent upper and lower bound functions on the prediction error
    uncertainty_neurons: list, optional
        List containing two integers, corresponding to the number of neurons in the first and last hidden layers of the neural network used to approximate the input-dependent uncertainty-bound functions. The neural network has the following structure:

                    v1 = W1 @ x + b1
                    v2 = W2 @ activation(v1) + b2
                    sigma = W3 @ sigmoid(v2) + b3

        If None, a uniform uncertainty bound is used. Only used if uncertainty is 'variable-symmetric' or 'variable-asymmetric'.
    uncertainty_activation: function
        Activation function used in the neural network to approximate the input-dependent uncertainty bounds.
    alpha: float
        Weight on amplitude of the uncertainty bound in the loss function to train the neural network to approximate the input-dependent uncertainty bounds.
    gamma_eps: float
        Smoothing parameter used in the loss function to train the neural network to approximate the input-dependent uncertainty bounds. Large values of gamma_eps lead to a better approximation of the maximum error, but also to a more difficult optimization problem.
    rho: float
        L2-regularization parameter for the neural network used to approximate the input-dependent uncertainty bounds.
        Large values of rho may lead to overfitting the uncertainty bound function on the training data, which later causes a larger scaling of the bound function to cover the error on the entire input set.
    seeds: list or None
        Seeds for parallel training. If None, all available cores are used.
    iprint: int
        Print level for the model training. If -1, no output is printed.
    global_optimizer: str
        Global optimization method to be used. See solve_global_optimization() for details.

    Returns:
    -----------
    model_epsil_u: StaticModel
        Upper-bound uncertainty function on the prediction error.
    model_epsil_ell: StaticModel
        Lower-bound uncertainty function on the prediction error.
    """

    if lb is None:
        lb = np.min(X, axis=0)
    if ub is None:
        ub = np.max(X, axis=0)
    if seeds is None:
        seeds = range(cpu_count())

    nx = X[0].size
    ny = 1
    Y = np.array(Y).reshape(-1, ny)
    N = Y.shape[0]

    def constant_uncertainty_fcn(x, params):
        return np.ones((x.shape[0], 1)) * params[0]

    params = model.params
    def fhat(x):
        return model.output_fcn(x.reshape(1, -1), params)

    if uncertainty == 'constant-symmetric':
        model_epsil_u = StaticModel(ny, nx, output_fcn=constant_uncertainty_fcn)
        model_epsil_u.init(np.array([worst_error]))
        model_epsil_ell = deepcopy(model_epsil_u)

    elif uncertainty == 'constant-asymmetric':
        def globopt_loss_u(x):
            return (-np.maximum(fun(x) - fhat(x),0.)).item()
        _, f_opt = solve_global_optimization(globopt_loss_u, lb, ub, global_optimizer)
        model_epsil_u = StaticModel(ny, nx, output_fcn=constant_uncertainty_fcn)
        model_epsil_u.init(np.array([-f_opt]))

        def globopt_loss_ell(x):
            return (-np.maximum(fhat(x) - fun(x),0.)).item()
        _, f_opt = solve_global_optimization(globopt_loss_ell, lb, ub, global_optimizer)
        model_epsil_ell = StaticModel(ny, nx, output_fcn=constant_uncertainty_fcn)
        model_epsil_ell.init(np.array([-f_opt]))

    elif uncertainty in ('variable-symmetric', 'variable-asymmetric'):
        jax.config.update('jax_platform_name', 'cpu')
        if not jax.config.jax_enable_x64:
            jax.config.update("jax_enable_x64", True)
            
        Yhat = model.predict(X).reshape(-1, ny)
        E = (Y - Yhat).reshape(-1)

        @jax.jit
        def sigma_fcn_model(u, params):
            W1, b1, W2, b2, W3, b3 = params
            y = W1 @ u.reshape(-1, 1) + b1
            y = W2 @ uncertainty_activation(y) + b2
            y = W3 @ nn.sigmoid(y) + b3
            return y.reshape(-1)
        sigma_fcn = jax.jit(jax.vmap(sigma_fcn_model, in_axes=(0, None)))

        def init_fcn_sigma(seed):
            np.random.seed(seed)
            return [np.random.randn(uncertainty_neurons[0], nx), np.random.randn(uncertainty_neurons[0], 1),
                    np.random.randn(uncertainty_neurons[1], uncertainty_neurons[0]), np.random.randn(uncertainty_neurons[1], 1),
                    np.random.rand(ny, uncertainty_neurons[1]), np.random.rand(ny, 1)]

        sigma_params_min = [-np.inf*np.ones((uncertainty_neurons[0], nx)), -np.inf*np.ones((uncertainty_neurons[0], 1)),
                            -np.inf*np.ones((uncertainty_neurons[1], uncertainty_neurons[0])), -np.inf*np.ones((uncertainty_neurons[1], 1)),
                            np.zeros((ny, uncertainty_neurons[1])), np.zeros((ny, 1))]

        def mu_weight(Ehat):
            return jnp.mean(alpha*Ehat**2)

        if uncertainty == 'variable-symmetric':
            @jax.jit
            def envelope_loss(Ehat, E):
                loss = jax.nn.logsumexp(jnp.hstack((0.,gamma_eps*(E-Ehat).reshape(-1))))/gamma_eps
                loss += mu_weight(Ehat)
                return loss

            def fit_quality(_, Ehat):
                return -mu_weight(Ehat[:, 0])

            sigma_s = np.abs(E) # absolute value of the error is used as target for the symmetric error model
            model_sigma = StaticModel(ny, nx, output_fcn=sigma_fcn)
            model_sigma.loss(rho_th=rho, output_loss=envelope_loss)
            model_sigma.optimization(adam_epochs=adam_epochs, lbfgs_epochs=lbfgs_epochs, params_min=sigma_params_min, iprint=iprint)
            models = model_sigma.parallel_fit(sigma_s, X, init_fcn=init_fcn_sigma, seeds=range(cpu_count()))
            model_epsil_u_unclipped, _ = find_best_model(models, sigma_s, X, fit=fit_quality, verbose=False)

            def kappa_loss(x):
                kappa = np.abs(fun(x) - fhat(x)) / model_epsil_u_unclipped.predict(x.reshape(1, -1))
                return -kappa.item()
            _, f_opt = solve_global_optimization(kappa_loss, lb, ub, global_optimizer)
            _scale_last_layer(model_epsil_u_unclipped, -f_opt)

            @jax.jit
            def clipped_output_fcn(x, _):
                return jnp.minimum(model_epsil_u_unclipped.predict(x), worst_error)
            model_epsil_u = StaticModel(ny, nx, output_fcn=clipped_output_fcn)
            model_epsil_u.init(model_epsil_u_unclipped.params)
            model_epsil_ell = deepcopy(model_epsil_u)

        else:  # 'variable-asymmetric'
            n_params = len(uncertainty_neurons)*2 + 2

            def combined_output_fcn(u, params):
                sigma_u = sigma_fcn(u, params[0:n_params])
                sigma_ell = sigma_fcn(u, params[n_params:])
                return jnp.concatenate((sigma_u, sigma_ell), axis=1)

            def init_fcn_combined(seed):
                np.random.seed(seed)
                return init_fcn_sigma(seed) + init_fcn_sigma(2*seed)

            @jax.jit
            def combined_loss(Sigma, E):
                loss = mu_weight(Sigma[:, 0]+Sigma[:, 1])
                loss += jax.nn.logsumexp(gamma_eps*jnp.hstack((E[:, 0]-Sigma[:, 0], 0., -E[:, 0]-Sigma[:, 1])))/gamma_eps
                return loss

            def fit_quality(_, Sigma):
                return -mu_weight(Sigma[:, 0]+Sigma[:, 1])

            model_sigma = StaticModel(2*ny, nx, output_fcn=combined_output_fcn)
            model_sigma.loss(rho_th=rho, output_loss=combined_loss)
            model_sigma.optimization(adam_epochs=adam_epochs, lbfgs_epochs=lbfgs_epochs,
                                     params_min=sigma_params_min + sigma_params_min, iprint=iprint)
            Y_combined = np.hstack((E.reshape(-1, 1), np.zeros((N, 1))))
            models = model_sigma.parallel_fit(Y_combined, X, init_fcn=init_fcn_combined, seeds=range(cpu_count()))
            model_sigma_combined, _ = find_best_model(models, E, X, fit=fit_quality, verbose=False)

            model_epsil_u_unclipped = StaticModel(ny, nx, output_fcn=sigma_fcn)
            model_epsil_u_unclipped.init(model_sigma_combined.params[0:n_params])
            model_epsil_ell_unclipped = StaticModel(ny, nx, output_fcn=sigma_fcn)
            model_epsil_ell_unclipped.init(model_sigma_combined.params[n_params:])

            def kappa_loss_u(x):
                kappa = np.maximum(fun(x) - fhat(x), 0.) / model_epsil_u_unclipped.predict(x.reshape(1, -1))
                return -kappa.item()
            _, f_opt = solve_global_optimization(kappa_loss_u, lb, ub, global_optimizer)
            _scale_last_layer(model_epsil_u_unclipped, -f_opt)

            def kappa_loss_ell(x):
                kappa = np.maximum(fhat(x) - fun(x), 0.) / model_epsil_ell_unclipped.predict(x.reshape(1, -1))
                return -kappa.item()
            _, f_opt = solve_global_optimization(kappa_loss_ell, lb, ub, global_optimizer)
            _scale_last_layer(model_epsil_ell_unclipped, -f_opt)

            @jax.jit
            def clipped_output_fcn_u(x, _):
                return jnp.minimum(model_epsil_u_unclipped.predict(x), worst_error)
            model_epsil_u = StaticModel(ny, nx, output_fcn=clipped_output_fcn_u)
            model_epsil_u.init(model_epsil_u_unclipped.params)

            @jax.jit
            def clipped_output_fcn_ell(x, _):
                return jnp.minimum(model_epsil_ell_unclipped.predict(x), worst_error)
            model_epsil_ell = StaticModel(ny, nx, output_fcn=clipped_output_fcn_ell)
            model_epsil_ell.init(model_epsil_ell_unclipped.params)

    else:
        raise ValueError(f'Unknown uncertainty type "{uncertainty}"')

    return model_epsil_u, model_epsil_ell

@jax.jit
def _mse_loss(Yhat, Y):
    return jnp.sum((Yhat-Y)**2)/Y.shape[0]

@jax.jit
def _minus_max_loss(Y, Yhat):
    return -jnp.max(jnp.abs(Y-Yhat))

def _train_cold_start(trained_model, X, Y, init_fcn, model_init, output_loss):
    if model_init == 'soft_max':
        trained_model.output_loss = output_loss
        models = trained_model.parallel_fit(Y, X, init_fcn=init_fcn, seeds=range(cpu_count()), n_jobs=cpu_count())
        trained_model, _ = find_best_model(models, Y, X, fit=_minus_max_loss, verbose=False)

    elif model_init == 'mse':
        trained_model.output_loss = _mse_loss
        models = trained_model.parallel_fit(Y, X, init_fcn=init_fcn, seeds=range(cpu_count()), n_jobs=cpu_count())
        params = [m.params for m in models]
        trained_model.output_loss = output_loss
        models = trained_model.parallel_fit(Y, X, init_fcn=lambda seed: params[seed], seeds=range(cpu_count()), n_jobs=cpu_count())
        trained_model, _ = find_best_model(models, Y, X, fit=_minus_max_loss, verbose=False)

    elif model_init == 'no':
        trained_model.params = init_fcn(0)

    else:
        raise ValueError(f'Unknown model_init option "{model_init}"')

    return trained_model


def maxfit(model, fun, X, init_fcn, lb=None, ub=None,
           method='function', maxiter=50, max_error=None, gamma=100., nu=0.,
           seeds=None, N_act_repeat=10, model_init='mse',
           adam_epochs=None, lbfgs_epochs=None, rho_th=None, iprint=None, warm_retrain=False, 
           global_optimizer='direct', eta=10.):
    """
    Approximates a given function using active learning of corner-case samples via global optimization.

    model: StaticModel
        Model to be trained. This is a StaticModel object from the jax_sysid library.
    fun: function
        Function to be approximated. This must have the form y=fun(x) where x is a vector of inputs and y is a scalar or vector output.
    X: array
        Initial set of feature vectors.
    init_fcn: function
        A function that initializes the model parameters given a seed.
    lb: array or None
        Lower bounds for the input variables.
    ub: array or None
        Upper bounds for the input variables.
    method: str
        'function': worst-case regression problem
        'set': worst-case set-approximation problem, used to approximate the set fun(x)<=0 with a surrogate fhat(x)<=0.
    maxiter: int
        Maximum number of active-learning iterations.
    max_error: float
        Maximum error threshold for stopping the active-learning process. If None, the process is stopped after maxiter iterations regardless of the error.
    gamma: float
        Soft-maximum parameter.
    nu: float
        Weight on MSE term added on top of the soft-maximum loss.
    seeds: list or None
        Seeds for initial parallel training
    N_act_repeat: int
        Number of times the new sample is repeated in the dataset after each active-learning step.
        Equivalent to increase the weight of the new sample during training.
    model_init: string
        'soft_max' = the model is initialized using the soft_max + nu*mse loss
        'mse' = the model is initialized using MSE loss, then refined by soft_max loss
        'no' = no initialization, the model is assumed already initialized.
    adam_epochs: int or None
        Number of Adam iterations during model retraining (default: same as iterations as specified in "model").
        At the first iteration (k=0), the model is always retrained for the number of epochs specified in "model", regardless of the value of adam_epochs.
    lbfgs_epochs: int or None
        Number of L-BFGS iterations during model retraining (default: same as iterations as specified in "model").
    rho_th: float or None
        L2-regularization during model retraining (default: same as iterations as specified in "model").
    iprint: int or None
        Print level for the model retraining phase. If -1, no output is printed (default: same as iterations as specified in "model").
    warm_retrain: bool
        If False, the model is retrained from scratch after each active-learning step. Otherwise, the model is retrained using the previous model parameters as initialization.
    global_optimizer: str
        Global optimization method to be used. See solve_global_optimization() for details.
    eta: float
        Parameter used to define the soft-sign function tanh(eta*y) used to approximate the sign function in method='set'.

    Returns:
        best_model: StaticModel
            The best model found during the active-learning process.
        results: SimpleNamespace with the following attributes:
            MaxErr: list
                List containing the maximum error at each iteration.
            MSErr: list
                List containing the MSE error of the best model at each iteration.
            X: array
                Feature vectors (initial + actively learned).
            Y: array
                Output values corresponding to X.
            N_act: int
                Number of actively learned samples.
            worst_error: float
                The worst-case error found during the active-learning process (method='function' only).
            time_training: float
                Total time spent in the model training phase during the active-learning process.
            time_worst_case: float
                Total time spent in the worst-case optimization phase (global optimization) during the active-learning process.
    """

    if lb is None:
        lb = np.min(X, axis=0)
    if ub is None:
        ub = np.max(X, axis=0)
    if seeds is None:
        seeds = range(cpu_count())

    if method == 'function':
        true_fun = fun
        trained_model = model
    elif method == 'set':
        def true_fun(x):
            return np.tanh(eta*fun(x))
        trained_model = deepcopy(model)
        @jax.jit
        def sign_output(x, params):
            y = model.output_fcn(x, params)
            return jnp.tanh(eta*y)
        trained_model.output_fcn = sign_output
    else:
        raise ValueError(f'Unknown method {method}')

    N0, nx = X.shape
    Y = np.array([true_fun(X[i, :]) for i in range(N0)]).reshape(N0, -1) # get output data corresponding to X
    ny = Y.shape[1]

    best_model = deepcopy(trained_model)
    best_error = np.inf
    best_mse = np.inf
    k = 0
    i = 0

    launch_char = "\N{ROCKET}"
    new_best_char = "\N{CLAPPING HANDS SIGN}"
    not_new_best_char = "\N{WASTEBASKET}"

    @jax.jit
    def soft_max_loss(Yhat, Y):
        gamma_E = gamma*(Y-Yhat)
        return jax.nn.logsumexp(jnp.vstack((gamma_E, -gamma_E)))/gamma

    @jax.jit
    def soft_max_mse_loss(Yhat, Y):
        return soft_max_loss(Yhat, Y) + nu*_mse_loss(Yhat, Y)

    MaxErr = []
    MSErr = []
    X_act = np.zeros((0, nx))
    Y_act = np.zeros((0, ny))
    N_act = 0

    time_training = 0.
    time_worst_case = 0.

    adam_epochs = model.adam_epochs if adam_epochs is None else adam_epochs
    lbfgs_epochs = model.lbfgs_epochs if lbfgs_epochs is None else lbfgs_epochs
    rho_th = model.rho_th if rho_th is None else rho_th
    iprint = model.iprint if iprint is None else iprint
    
    print("Fitting initial model ... ", end="")
    
    while k < maxiter and (max_error is None or best_error > max_error):

        output_loss = soft_max_mse_loss
        start_time = time.time()

        if k==1:
            # Change training parameters if specified by the user, otherwise keep the same as in the previous iteration
            trained_model.adam_epochs = adam_epochs
            trained_model.lbfgs_epochs = lbfgs_epochs
            trained_model.rho_th = rho_th
            trained_model.iprint = iprint

        if k == 0 or not warm_retrain:
            trained_model = _train_cold_start(trained_model, X, Y, init_fcn,
                                              model_init if k == 0 else 'soft_max', output_loss)
        else:
            trained_model.output_loss = output_loss
            trained_model.fit(Y, X)

        time_training += time.time() - start_time

        params = trained_model.params
        def fhat(x):
            return trained_model.output_fcn(x.reshape(1, -1), params)

        def globopt_loss(x):
            return (-np.max(np.abs(true_fun(x) - fhat(x)))).item()

        start_time = time.time()
        x_new, f_opt = solve_global_optimization(globopt_loss, lb, ub, global_optimizer)
        time_worst_case += time.time() - start_time

        N_act += 1

        current_max_err = -f_opt

        if current_max_err <= best_error:
            best_error = current_max_err
            best_model = deepcopy(trained_model)
            best_mse = np.sum((trained_model.predict(X) - Y)**2)/Y.shape[0]
            i = 0

        MaxErr.append(best_error)
        MSErr.append(best_mse)

        X_act = np.vstack((X_act, x_new.reshape(-1, nx)))
        y_new = true_fun(x_new.reshape(-1, nx)).reshape(-1, ny)
        Y_act = np.vstack((Y_act, y_new))
        for _ in range(N_act_repeat):
            X = np.vstack((X, x_new.reshape(-1, nx)))
            Y = np.vstack((Y, y_new))

        if k == 0:
            print("done.")
        icon = launch_char if k == 0 else (new_best_char if i == 0 else not_new_best_char)
        print(f"k = {k+1: 3d}/{maxiter}: max error = {current_max_err: 10.6f} (best = {best_error: 10.6f}), RMSE = {np.sqrt(best_mse): 10.8f} {icon}")

        k += 1
        i += 1

    if method == 'function':
        output_model = best_model

    else:
        def sign(x):
            return -1.0 if x <= 0 else 1.0

        tmp_model = deepcopy(model)
        tmp_model.params = best_model.params

        def eps_loss(x):
            return np.array(.5*(1.+sign(true_fun(x)))*tmp_model.predict(x.reshape(1, -1))).item()

        start_time = time.time()
        _, delta = solve_global_optimization(eps_loss, lb, ub, global_optimizer)
        time_worst_case += time.time() - start_time

        output_model = deepcopy(model)
        @jax.jit
        def output_fcn_with_delta(x, params):
            return model.output_fcn(x, params[:-1]) - params[-1]
        output_model.output_fcn = output_fcn_with_delta
        output_model.params = best_model.params + [delta]
        best_error = delta

    worst_error = best_error

    results = SimpleNamespace()
    results.MaxErr = MaxErr
    results.MSErr = MSErr
    results.X = np.vstack((X[:N0, :], X_act))
    results.Y = np.vstack((Y[:N0, :], Y_act)).reshape(-1)
    if method == 'set':
        results.Y = fun(results.X)
    results.worst_error = worst_error
    results.N_act = N_act
    results.time_training = time_training
    results.time_worst_case = time_worst_case
    return output_model, results
