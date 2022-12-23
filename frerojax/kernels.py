import jax
import jax.numpy as jnp

Array = jnp.ndarray

def squared_pdist(x: Array, y: Array):
    return jax.vmap(lambda xx: jax.vmap(lambda yy: jnp.clip(xx@xx + yy@yy - 2.0*xx@yy, a_min=0))(y))(x)

def RBF(x: Array, y: Array, lengthscale: float) -> float:
    sqdist = squared_pdist(x, y)
    return jnp.exp(-0.5*sqdist/lengthscale**2)