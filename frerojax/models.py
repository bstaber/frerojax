import jax
import jax.numpy as jnp
import jax.scipy as scipy

from typing import NamedTuple, Callable
from frerojax import likelihoods

Array = jnp.ndarray

class BaseGPR(NamedTuple):
    loglike_fn: Callable
    alpha_fn: Callable
    predict_fn: Callable
    
class EuclideanGPR:
    
    """Implements the basic elements for Gaussian process regression (without noisy observations).
    
    Example
    -------
    
    Assume we have some training data :math:`(X_train, y_train)` and a chosen kernel function.
    
    .. code::
        
        from frerojax.kernels import RBF as kernel_fn
        
        model = EuclideanGPR(kernel_fn, X_train, y_train)

    The hyperparameters of the kernel function are determined by maximizing the marginal log-likelihood.
    In this example, we use a simple Gaussian kernel with a single hyperparameter (the lengthscale).
    
    .. code::
    
        def obj_fn(params):
            return model.loglike_fn(jnp.sqrt(jnp.exp(-params)))
        results = optimize.minimize(obj_fn, jnp.array([0.0]), method="BFGS")

    Get the solution and make predictions
    
    .. code ::
    
        lengthscale = jnp.sqrt(jnp.exp(-results.x[0]))
        alpha_array = model.alpha_fn(lengthscale)

        x_new = jnp.linspace(-3, 3, 100)[:,None]
        y_true = target_fn(x_new)
        y_new = model.predict_fn(lengthscale, alpha_array, x_new)

    """
    
    marginal_loglike = likelihoods.marginal_loglike
    
    def __new__(
        cls,
        kernel_fn: Callable,
        X_train: Array,
        y_train: Array
        ):
        
        def loglike_fn(parameters):
            return cls.marginal_loglike(parameters, kernel_fn, X_train, y_train)
        
        def alpha_fn(parameters):
            Kxx = kernel_fn(X_train, X_train, parameters)
            chol = scipy.linalg.cho_factor(Kxx)
            alpha_pred = scipy.linalg.cho_solve(chol, y_train)
            return alpha_pred
        
        def predict_fn(parameters, alpha_array: Array, X_test: Array):
            Ktx = kernel_fn(X_test, X_train, parameters)
            f_pred = jnp.matmul(Ktx, alpha_array)
            return f_pred
        
        return BaseGPR(loglike_fn=loglike_fn, alpha_fn=alpha_fn, predict_fn=predict_fn)