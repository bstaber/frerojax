import jax
import jax.numpy as jnp
import jax.scipy as scipy
from typing import Callable, Union

Array = jnp.array

def marginal_loglike(parameters: Union[float,dict], kernel_fn: Callable, X_train: Array, y_train: Array) -> float:
    """Computes the marginal log-likelihood of the conditioned Gaussian process.
    """
    Kxx = kernel_fn(X_train, X_train, parameters)
    chol = scipy.linalg.cho_factor(Kxx)
    alpha_pred = scipy.linalg.cho_solve(chol, y_train)
    return - 0.5*jnp.dot(y_train, alpha_pred) - jnp.sum(jnp.diag(jnp.log(chol[0])))