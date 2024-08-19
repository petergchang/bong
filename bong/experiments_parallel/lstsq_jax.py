import jax.numpy as jnp
import jax.random as jr
import jax
import subprocess


def jax_has_gpu():
    # https://github.com/google/jax/issues/971
    try:
        _ = jax.device_put(jax.numpy.ones(1), device=jax.devices("gpu")[0])
        return True
    except:
        return False


def get_gpu_name():
    if not jax_has_gpu():
        return "None"
    # Run the nvidia-smi command
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    name = result.stdout.strip()
    return name


print("Using jax version ", jax.__version__)

print("Using GPU ", get_gpu_name())

key = jr.PRNGKey(0)
N, D = 500, 22
key, subkey = jr.split(key)
X = jr.normal(subkey, shape=(N, D))
key, subkey = jr.split(key)
y = jr.normal(subkey, shape=(N,))

params, residuals, rank, s = jnp.linalg.lstsq(X, y, rcond=None)
print(params)
