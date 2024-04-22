from typing import Any

import jax
import jax.random as jr
import jax_tqdm

from bong.base import RebayesAlgorithm, State
from bong.types import ArrayLike, PRNGKey


def run_rebayes_algorithm(
    rng_key: PRNGKey,
    rebayes_algorithm: RebayesAlgorithm,
    X: ArrayLike,
    Y: ArrayLike,
    init_state: State=None,
    transform=lambda state, x, y: state,
    progress_bar: bool = False,
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
    
    Returns:
        Final belief state and extra information.
    """
    num_timesteps = len(X)
    if init_state is None:
        init_state = rebayes_algorithm.init(**init_kwargs)
    
    @jax.jit
    def _step(state, args):
        key, t = args
        x, y = X[t], Y[t]
        pred_state = rebayes_algorithm.predict(state)
        output = transform(pred_state, x, y)
        new_state = rebayes_algorithm.update(key, pred_state, x, y)
        return new_state, output
    
    if progress_bar:
        _step = jax_tqdm.scan_tqdm(num_timesteps)(_step)
    
    keys = jr.split(rng_key, num_timesteps)
    args = (keys, jax.numpy.arange(num_timesteps))
    final_state, outputs = jax.lax.scan(_step, init_state, args)
    return final_state, outputs