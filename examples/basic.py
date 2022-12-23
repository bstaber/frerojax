import jax
import jax.numpy as jnp
import jax.scipy.optimize as optimize

from frerojax.models import EuclideanGPR
from frerojax.kernels import RBF as kernel_fn

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use("seaborn")

# Dummy dataset
target_fn = lambda x: jnp.cos(3.0*x) + jnp.sin(x)

rng_key = jax.random.PRNGKey(0)
X_train = jnp.linspace(-3, 3, 12)[:,None]
y_train = target_fn(X_train).squeeze()

# Model (Gaussian process for tabular data)
model = EuclideanGPR(kernel_fn, X_train, y_train)

# Define objective function and call BFGS
def obj_fn(params):
    return -model.loglike_fn(jnp.sqrt(jnp.exp(-params)))
results = optimize.minimize(obj_fn, jnp.array([0.0]), method="BFGS")
print(results)

# Predict
lengthscale = jnp.sqrt(jnp.exp(-results.x[0]))
print(f"Solution: {lengthscale}")
alpha_array = model.alpha_fn(lengthscale)

x_new = jnp.linspace(-3, 3, 100)[:,None]
y_true = target_fn(x_new)
y_new = model.predict_fn(lengthscale, alpha_array, x_new)

# Plot
fig, ax = plt.subplots(1,1,figsize=(6,6))
ax.plot(X_train, y_train, linestyle="", color="b", marker="o", markeredgecolor="k", markeredgewidth=1.0, label="Training data")
ax.plot(x_new, y_true, linestyle="-", color="k", linewidth=2, label="True function")
ax.plot(x_new, y_new, linestyle="--", color="r", linewidth=2, label="Prediction")
ax.tick_params(labelsize=12)
ax.legend(fontsize=14)
fig.savefig("basic.png", format="png")