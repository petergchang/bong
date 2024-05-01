from typing import Any, Callable, Sequence

from flax import linen as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import jax_tqdm
import optax
import optuna

from bong.base import RebayesAlgorithm, State
from bong.types import Array, ArrayLike, PRNGKey


class MLP(nn.Module):
    features: Sequence[int]
    activation: nn.Module = nn.relu
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        x = x.ravel()
        for feat in self.features[:-1]:
            x = self.activation(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1], use_bias=self.use_bias)(x)
        return x


def hess_diag_approx(
    rng_key: PRNGKey,
    fn: Callable,
    param: ArrayLike,
    num_samples: int = 100,
) -> Array:
    """Approximate the diagonal of the Hessian of a function using the
    Hutchinson's method 
        ref: equation (9) of https://arxiv.org/pdf/2006.00719.pdf

    Args:
        rng_key: JAX PRNG Key.
        fn: Function to compute the Hessian of.
        param: Parameters to compute the Hessian at.
        num_samples: Number of samples to use for the approximation.

    Returns:
        Approximate diagonal of the Hessian.
    """
    def _hess_diag(z):
        return z * jax.grad(lambda p: jax.grad(fn)(p) @ z)(param)
    zs = jr.rademacher(rng_key, (num_samples, len(param)))
    return jnp.mean(jax.vmap(_hess_diag)(zs), axis=0)


def nearest_psd_matrix(mat, eps=1e-6):
    eigenvalues, eigenvectors = jnp.linalg.eigh(mat)
    eigenvalues = jnp.maximum(eigenvalues, eps)
    return (eigenvectors @ jnp.diag(eigenvalues)) @ eigenvectors.T


def run_rebayes_algorithm(
    rng_key: PRNGKey,
    rebayes_algorithm: RebayesAlgorithm,
    X: ArrayLike,
    Y: ArrayLike,
    init_state: State=None,
    transform=lambda key, alg, state, x, y: state,
    progress_bar: bool=False,
    n_iter: int=None,
    **init_kwargs,
) -> tuple[State, Any]:
    """Run a rebayes algorithm over a sequence of observations.
    
    Args:
        rng_key: JAX PRNG Key.
        rebayes_algorithm: Rebayes algorithm to run.
        X: Sequence of inputs.
        Y: Sequence of outputs.
        init_state: Initial belief state.
        transform: Transform the belief state after each update.
        progress_bar: Whether to display a progress bar.
        n_iter: Number of iterations to run the algorithm for.
    
    Returns:
        Final belief state and extra information.
    """
    num_timesteps = len(X)
    if init_state is None:
        init_state = rebayes_algorithm.init(**init_kwargs)
    
    @jax.jit
    def _step(state, t):
        key, subkey = jr.split(jr.fold_in(rng_key, t))
        x, y = X[t], Y[t]
        pred_state = rebayes_algorithm.predict(state)
        output = transform(key, rebayes_algorithm, pred_state, x, y)
        new_state = rebayes_algorithm.update(subkey, pred_state, x, y)
        return new_state, output
    
    if progress_bar:
        _step = jax_tqdm.scan_tqdm(num_timesteps)(_step)
    
    args = jnp.arange(num_timesteps)
    final_state, outputs = jax.lax.scan(_step, init_state, args)
    return final_state, outputs


def tune_init_cov(
    rng_key: PRNGKey,
    rebayes_algorithm_initializer: Any,
    X: ArrayLike,
    Y: ArrayLike,
    loss_fn: Callable,
    n_trials=20,
    minval=-10.0,
    maxval=0.0,
    **init_kwargs,
):
    def _objective(trial):
        log_init_cov = trial.suggest_float("log_init_cov", minval, maxval)
        init_cov = jnp.exp(log_init_cov).item()
        rebayes_algorithm = rebayes_algorithm_initializer(
            init_cov=init_cov,
            **init_kwargs,
        )
        key, subkey = jr.split(rng_key)
        state, _ = run_rebayes_algorithm(
            key, rebayes_algorithm, X, Y,
        )
        eval_loss = loss_fn(subkey, rebayes_algorithm, state)
        return eval_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(_objective, n_trials=n_trials)
    
    best_trial = study.best_trial
    best_params = best_trial.params
    return best_params
    