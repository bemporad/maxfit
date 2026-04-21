<img src="http://cse.lab.imtlucca.it/~bemporad/maxfit/images/maxfit-logo.png" alt="nashopt" width=40%/>

# MaxFit  
### A Python library for Worst-Case Nonlinear Regression

This repository includes a library for solving worst-case nonlinear regression problems.
Given the function to approximate, the library attempts fitting a model that minimizes the worst-case error over a given set. Optionally, it can compute upper and lower bound functions on the resulting error.

For more details about the mathematical formulations implemented in the library, see the 
<a href="https://arxiv.org/abs/2601.12334">arXiv preprint 2601.12334</a>.

---
## Installation

~~~python
pip install maxfit
~~~


## Overview

**MaxFit** trains a nonlinear surrogate model $\hat f(x)$ to minimize the **worst-case approximation error**

$$e^* = \max_{x \in [l_b,\, u_b]} |f(x) - \hat f(x)|$$

rather than the usual mean-squared error. The algorithm alternates between:

1. **Training** the surrogate on the current dataset using a soft-maximum loss function.
2. **Active learning**: finding the input $x^*$ where the current error is largest via global optimization.
3. **Augmenting** the dataset with $x^*$ and repeating until $e^*$ falls below a tolerance or a maximum number of iterations is reached.

The library provides two main functions:

- **`maxfit(model, fun, X, init_fcn, lb, ub, ...)`** — runs the active-learning loop and returns the best surrogate model together with convergence statistics. Also supports **set approximation**: fitting a surrogate $\hat f(x) \leq 0$ to the set $\{x : f(x) \leq 0\}$.

- **`uncertainty_bounds(model, fun, X, Y, lb, ub, ...)`** — fits input-dependent error envelopes $[\hat f(x) - \varepsilon_\ell(x),\; \hat f(x) + \varepsilon_u(x)]$ that contain $f(x)$ for all $x \in [l_b, u_b]$, using either constant or neural-network-parameterized bound functions. The envelopes are certified bounds, assuming the used global optimizer is able to determine the worst-case errors.

Surrogate models are [**jax-sysid**](https://github.com/bemporad/jax-sysid) `StaticModel` objects, so any differentiable JAX architecture can be used. Global optimization is performed via the [**NLopt**](https://nlopt.readthedocs.io) library, with **DIRECT** as default optimizer.


## Example

Approximate the scalar function $f(x) = \frac{(\sin(x(1-x/10)) + (x/10)^3 - 4x/10)\,e^{-x}}{1+e^{-x}}$ on $[-10, 10]$ starting from 20 random samples, then fit asymmetric input-dependent uncertainty envelopes.

```python
import numpy as np
from maxfit import maxfit, uncertainty_bounds
from jax_sysid.models import StaticModel
import jax
from flax import linen as nn

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

def fun(x):
    x0 = x[0]
    return (np.sin(x0*(1.-x0/10.))+(x0/10.)**3-4.*x0/10.)*np.exp(-x0)/(1.+np.exp(-x0))

# Initial dataset
np.random.seed(0)
N = 20
nx, n1, n2 = 1, 2, 1
xmin, xmax = -10., 10.
X = np.random.uniform(xmin, xmax, N).reshape(-1, 1)
lb, ub = np.array([xmin]), np.array([xmax])

# Neural network model (residual connections, tanh activation)
act = nn.tanh

@jax.jit
def output_fcn_(u, params):
    V1, b1, W2, V2, b2, W3, V3, b3 = params
    u = u.reshape(-1, 1)
    y = W2 @ act(V1 @ u + b1) + V2 @ u + b2
    return (W3 @ act(y) + V3 @ u + b3).reshape(-1)
output_fcn = jax.jit(jax.vmap(output_fcn_, in_axes=(0, None)))

rho_th = 1.e-8
model = StaticModel(1, nx, output_fcn)
model.optimization(adam_epochs=1000, lbfgs_epochs=1000, iprint=-1)
model.loss(rho_th=rho_th, tau_th=0.)

def init_fcn(seed):
    np.random.seed(seed)
    rn = np.random.randn
    return [rn(n1, nx), rn(n1, 1), rn(n2, n1), rn(n2, nx), rn(n2, 1),
            rn(1, n2), rn(1, nx), np.random.rand(1, 1)]

# Worst-case regression
model, results = maxfit(model, fun, X, init_fcn, lb=lb, ub=ub, method='function',
                        maxiter=30, gamma=10., nu=0., model_init='mse',
                        warm_retrain=False, adam_epochs=1000, lbfgs_epochs=1000,
                        rho_th=rho_th, global_optimizer='direct')

print(f"Worst-case error: {results.worst_error:.6f}")
print(f"Actively acquired {results.N_act} corner-case samples")

# Input-dependent asymmetric uncertainty envelopes
x = np.linspace(xmin, xmax, 101).reshape(-1, 1)
y = np.array([fun(xi) for xi in x]).reshape(-1, 1)
X_eps = np.vstack((results.X, x))
Y_eps = np.vstack((results.Y.reshape(-1, 1), y))

model_eps_u, model_eps_ell = uncertainty_bounds(
    model, fun, X_eps, Y_eps, lb, ub,
    worst_error=results.worst_error,
    uncertainty='variable-asymmetric',
    uncertainty_neurons=[n1, n2],
    uncertainty_activation=act,
    rho=1.e-6, alpha=1.e-3, gamma_eps=100.,
    global_optimizer='direct',
    adam_epochs=2000, lbfgs_epochs=2000)
```

The full example including plots is available in [`examples/example_scalar_fun.py`](examples/example_scalar_fun.py). See [`examples/`](examples/) for additional examples.


## References

> [1] A. Bemporad, "[Worst-case Nonlinear Regression with Error Bounds](https://arxiv.org/abs/2601.12334)," arXiv preprint 2601.12334, 2026.

> [2] A. Bemporad, "[An L-BFGS-B approach for linear and nonlinear system identification under ℓ1 and group-Lasso regularization](https://doi.org/10.1109/TAC.2024.3406595)," *IEEE Transactions on Automatic Control*, vol. 70, no. 7, pp. 4857–4864, 2025. (**jax-sysid**)

> [3] D.R. Jones, M. Schonlau, and W.J. Matthias, "Efficient global optimization of expensive black-box functions," *Journal of Global Optimization*, vol. 13, no. 4, pp. 455–492, 1998. (**DIRECT global optimizer**)

## Citation

```
@article{MaxFit,
    author={A. Bemporad},
    title={Worst-case Nonlinear Regression with Error Bounds},
    journal = {arXiv preprint 2601.12334},
    note = {\url{https://github.com/bemporad/maxfit}},
    year=2026
}
```

---
## Related packages

<a href="https://github.com/bemporad/nash-mpqp">**jax-sysid**</a> a Python package based on JAX for linear and nonlinear system identification of state-space models, recurrent neural network (RNN) training, and nonlinear regression/classification

---
## License

Apache 2.0

(C) 2026 A. Bemporad

## Acknowledgement
This work was funded by the European Union (ERC Advanced Research Grant COMPACT, No. 101141351). Views and opinions expressed are however those of the authors only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them.

<p align="center">
<img src="erc-logo.png" alt="ERC" width="400"/>
</p>
