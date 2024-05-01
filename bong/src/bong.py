from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr

from bong.base import RebayesAlgorithm
from bong.types import ArrayLikeTree, ArrayTree, PRNGKey
from bong.util import hess_diag_approx


class BONGState(NamedTuple):
    """Belief state of a BONG agent.
    
    mean: Mean of the belief state.
    cov: Covariance of the belief state.
    """
    mean: ArrayTree
    cov: ArrayTree
    

def init_bong(
    init_mean: ArrayLikeTree,
    init_cov: ArrayLikeTree,
) -> BONGState:
    """Initialize the belief state with a mean and precision.
    
    Args:
        init_mean: Initial mean of the belief state.
        init_cov: Initial covariance of the belief state.
    
    Returns:
        Initial belief state.
    """
    return BONGState(mean=init_mean, cov=init_cov)


def predict_bong(
    state: BONGState,
    gamma: float,
    Q: ArrayLikeTree,
    *args,
    **kwargs,
) -> BONGState:
    """Predict the next state of the belief state.
    
    Args:
        state: Current belief state.
        gamma: Forgetting factor.
        Q: Process noise.
    
    Returns:
        Predicted belief state.
    """
    mean, cov = state
    new_mean = jax.tree_map(lambda x: gamma * x, mean)
    new_cov = jax.tree_map(lambda x, y: gamma**2 * x + y, cov, Q)
    new_state = BONGState(new_mean, new_cov)
    return new_state


def update_fg_bong(
    rng_key: PRNGKey,
    state: BONGState,
    x: ArrayLikeTree,
    y: ArrayLikeTree,
    log_likelihood: Callable,
    emission_mean_function: Callable,
    emission_cov_function: Callable,
    num_samples: int=10,
    empirical_fisher: bool=False,
    *args,
    **kwargs,
) -> BONGState:
    """Update the full-covariance Gaussian belief state with a new observation.
    
    Args:
        rng_key: JAX PRNG Key.
        state: Current belief state.
        x: Input.
        y: Observation.
        log_likelihood: Log-likelihood function.
        emission_mean_function: Emission mean function.
        emission_cov_function: Emission covariance function.
        num_samples: Number of samples to use for the update.
        empirical_fisher: Whether to use the empirical Fisher approximation
            to the Hessian matrix.
    
    Returns:
        Updated belief state.
    """
    mean, cov = state
    
    def ll_fn(param):
        emission_mean = emission_mean_function(param, x)
        emission_cov = emission_cov_function(param, x)
        return jnp.mean(log_likelihood(emission_mean, emission_cov, y))
    
    z = sample_fg_bong(rng_key, state, num_samples)
    grads = jax.vmap(jax.grad(ll_fn))(z)
    if empirical_fisher:
        prec_update = grads.T @ grads
    else:
        prec_update = jnp.mean(jax.vmap(jax.hessian(ll_fn))(z), axis=0)
    print(f"prec_update: {prec_update.shape}")
    prec = jnp.linalg.pinv(cov)
    new_prec = prec - prec_update
    new_cov = jnp.linalg.pinv(new_prec)
    mean_update = jnp.mean(grads, axis=0)
    new_mean = mean + new_cov @ mean_update
    new_state = BONGState(new_mean, new_cov)
    return new_state


def update_lfg_bong(
    rng_key: PRNGKey,
    state: BONGState,
    x: ArrayLikeTree,
    y: ArrayLikeTree,
    log_likelihood: Callable,
    emission_mean_function: Callable,
    emission_cov_function: Callable,
    *args,
    **kwargs,
) -> BONGState:
    """Update the linearized-plugin full-covariance Gaussian belief state
    with a new observation. Note that this is equivalent to the EKF.
    
    Args:
        rng_key: JAX PRNG Key.
        state: Current belief state.
        x: Input.
        y: Observation.
        log_likelihood: Log-likelihood function.
        emission_mean_function: Emission mean function.
        emission_cov_function: Emission covariance function.
    
    Returns:
        Updated belief state.
    """
    mean, cov = state
    y_pred = jnp.atleast_1d(emission_mean_function(mean, x))
    H = jnp.atleast_2d(jax.jacrev(emission_mean_function)(mean, x))
    R = jnp.atleast_2d(emission_cov_function(mean, x))
    S = R + (H @ cov @ H.T)
    C = cov @ H.T
    K = jnp.linalg.lstsq(S, C.T)[0].T
    new_mean = mean + K @ (y - y_pred)
    new_cov = cov - K @ S @ K.T
    new_state = BONGState(new_mean, new_cov)
    return new_state


def update_dg_bong(
    rng_key: PRNGKey,
    state: BONGState,
    x: ArrayLikeTree,
    y: ArrayLikeTree,
    log_likelihood: Callable,
    emission_mean_function: Callable,
    emission_cov_function: Callable,
    num_samples: int=10,
    *args,
    **kwargs,
) -> BONGState:
    """Update the diagonal-covariance Gaussian belief state with a new observation.
    
    Args:
        rng_key: JAX PRNG Key.
        state: Current belief state.
        x: Input.
        y: Observation.
        log_likelihood: Log-likelihood function.
        emission_mean_function: Emission mean function.
        emission_cov_function: Emission covariance function.
        num_samples: Number of samples to use for the update.
    
    Returns:
        Updated belief state.
    """
    mean, cov = state
    
    def ll_fn(param):
        emission_mean = emission_mean_function(param, x)
        emission_cov = emission_cov_function(param, x)
        return jnp.mean(log_likelihood(emission_mean, emission_cov, y))
    
    keys = jr.split(rng_key)
    z = sample_dg_bong(keys[0], state, num_samples)
    hess_diag_fn = lambda param: hess_diag_approx(keys[1], ll_fn, param)
    prec_update = jnp.mean(jax.vmap(hess_diag_fn)(z), axis=0)
    prec = 1 / cov
    new_prec = prec - prec_update
    new_cov = 1 / new_prec
    mean_update = jnp.mean(jax.vmap(jax.grad(ll_fn))(z), axis=0)
    new_mean = mean + new_cov * mean_update
    new_state = BONGState(new_mean, new_cov)
    return new_state


def update_ldg_bong(
    rng_key: PRNGKey,
    state: BONGState,
    x: ArrayLikeTree,
    y: ArrayLikeTree,
    log_likelihood: Callable,
    emission_mean_function: Callable,
    emission_cov_function: Callable,
    num_samples: int=10,
    *args,
    **kwargs,
) -> BONGState:
    """Update the linearized-plugin diagonal-covariance Gaussian 
    belief state with a new observation. Note that this is 
    equivalent to the VD-EKF.
    
    Args:
        rng_key: JAX PRNG Key.
        state: Current belief state.
        x: Input.
        y: Observation.
        log_likelihood: Log-likelihood function.
        emission_mean_function: Emission mean function.
        emission_cov_function: Emission covariance function.
        num_samples: Number of samples to use for the update.
    
    Returns:
        Updated belief state.
    """
    mean, cov = state
    y_pred = jnp.atleast_1d(emission_mean_function(mean, x))
    H = jnp.atleast_2d(jax.jacrev(emission_mean_function)(mean, x))
    R = jnp.atleast_2d(emission_cov_function(mean, x))
    K = jnp.linalg.lstsq((R + (cov * H) @ H.T).T, cov * H)[0].T
    R_inv = jnp.linalg.lstsq(R, jnp.eye(R.shape[0]))[0]
    new_cov = 1/(1/cov + ((H.T @ R_inv) * H.T).sum(-1))
    new_mean = mean + K @ (y - y_pred)
    new_state = BONGState(new_mean, new_cov)
    return new_state


def update_fg_reparam_bong(
    rng_key: PRNGKey,
    state: BONGState,
    x: ArrayLikeTree,
    y: ArrayLikeTree,
    log_likelihood: Callable,
    emission_mean_function: Callable,
    emission_cov_function: Callable,
    num_samples: int=10,
    empirical_fisher: bool=False,
    *args,
    **kwargs,
) -> BONGState:
    """Update the full-covariance Gaussian belief state with a new observation,
    under the reparameterized BONG model.
    
    Args:
        rng_key: JAX PRNG Key.
        state: Current belief state.
        x: Input.
        y: Observation.
        log_likelihood: Log-likelihood function.
        emission_mean_function: Emission mean function.
        emission_cov_function: Emission covariance function.
        num_samples: Number of samples to use for the update.
        empirical_fisher: Whether to use the empirical Fisher approximation
            to the Hessian matrix.
    
    Returns:
        Updated belief state.
    """
    mean, cov = state
    
    def ll_fn(param):
        emission_mean = emission_mean_function(param, x)
        emission_cov = emission_cov_function(param, x)
        return jnp.mean(log_likelihood(emission_mean, emission_cov, y))
    
    z = sample_fg_bong(rng_key, state, num_samples)
    grads = jax.vmap(jax.grad(ll_fn))(z)
    if empirical_fisher:
        cov_update = grads.T @ grads
    else:
        cov_update = jnp.mean(jax.vmap(jax.hessian(ll_fn))(z), axis=0)
    mean_update = jnp.mean(grads, axis=0)
    new_mean = mean + cov @ mean_update
    new_cov = cov + cov @ cov_update @ cov
    new_state = BONGState(new_mean, new_cov)
    return new_state


def update_lfg_reparam_bong(
    rng_key: PRNGKey,
    state: BONGState,
    x: ArrayLikeTree,
    y: ArrayLikeTree,
    log_likelihood: Callable,
    emission_mean_function: Callable,
    emission_cov_function: Callable,
    *args,
    **kwargs,
) -> BONGState:
    """Update the linearized-plugin full-covariance Gaussian belief state
    with a new observation under the reparameterized BONG model.
    
    Args:
        rng_key: JAX PRNG Key.
        state: Current belief state.
        x: Input.
        y: Observation.
        log_likelihood: Log-likelihood function.
        emission_mean_function: Emission mean function.
        emission_cov_function: Emission covariance function.
    
    Returns:
        Updated belief state.
    """
    mean, cov = state
    y_pred = jnp.atleast_1d(emission_mean_function(mean, x))
    H = jnp.atleast_2d(jax.jacrev(emission_mean_function)(mean, x))
    R = jnp.atleast_2d(emission_cov_function(mean, x))
    C = cov @ H.T
    K = jnp.linalg.lstsq(R, C.T)[0].T
    new_mean = mean + K @ (y - y_pred)
    new_cov = cov - K @ R @ K.T
    new_state = BONGState(new_mean, new_cov)
    return new_state


def update_dg_reparam_bong(
    rng_key: PRNGKey,
    state: BONGState,
    x: ArrayLikeTree,
    y: ArrayLikeTree,
    log_likelihood: Callable,
    emission_mean_function: Callable,
    emission_cov_function: Callable,
    num_samples: int=10,
    *args,
    **kwargs,
) -> BONGState:
    """Update the diagonal-covariance Gaussian belief state with a new observation
    under the reparameterized BONG model.
    
    Args:
        rng_key: JAX PRNG Key.
        state: Current belief state.
        x: Input.
        y: Observation.
        log_likelihood: Log-likelihood function.
        emission_mean_function: Emission mean function.
        emission_cov_function: Emission covariance function.
        num_samples: Number of samples to use for the update.
    
    Returns:
        Updated belief state.
    """
    mean, cov = state
    
    def ll_fn(param):
        emission_mean = emission_mean_function(param, x)
        emission_cov = emission_cov_function(param, x)
        return jnp.mean(log_likelihood(emission_mean, emission_cov, y))
    
    keys = jr.split(rng_key)
    z = sample_dg_bong(keys[0], state, num_samples)
    mean_update = jnp.mean(jax.vmap(jax.grad(ll_fn))(z), axis=0)
    new_mean = mean + cov * mean_update
    hess_diag_fn = lambda param: hess_diag_approx(keys[1], ll_fn, param)
    cov_update = jnp.mean(jax.vmap(hess_diag_fn)(z), axis=0)
    new_cov = cov + cov * cov_update * cov
    new_state = BONGState(new_mean, new_cov)
    return new_state


def update_ldg_reparam_bong(
    rng_key: PRNGKey,
    state: BONGState,
    x: ArrayLikeTree,
    y: ArrayLikeTree,
    log_likelihood: Callable,
    emission_mean_function: Callable,
    emission_cov_function: Callable,
    num_samples: int=10,
    *args,
    **kwargs,
) -> BONGState:
    """Update the linearized-plugin diagonal-covariance Gaussian 
    belief state with a new observation under the reparameterized BONG model.
    
    Args:
        rng_key: JAX PRNG Key.
        state: Current belief state.
        x: Input.
        y: Observation.
        log_likelihood: Log-likelihood function.
        emission_mean_function: Emission mean function.
        emission_cov_function: Emission covariance function.
        num_samples: Number of samples to use for the update.
    
    Returns:
        Updated belief state.
    """
    mean, cov = state
    y_pred = jnp.atleast_1d(emission_mean_function(mean, x))
    H = jnp.atleast_2d(jax.jacrev(emission_mean_function)(mean, x))
    R = jnp.atleast_2d(emission_cov_function(mean, x))
    HTRinv = jnp.linalg.lstsq(R, H)[0].T
    new_mean = mean + (cov * HTRinv.T).T @ (y - y_pred)
    new_cov = cov - cov**2 * (HTRinv * H.T).sum(-1)
    new_state = BONGState(new_mean, new_cov)
    return new_state


def sample_fg_bong(rng_key: PRNGKey, state: BONGState, num_samples: int=10):
    """Sample from the full-covariance Gaussian belief state"""
    mean, cov = state
    states = jr.multivariate_normal(rng_key, mean, cov, shape=(num_samples,))
    return states


def sample_dg_bong(rng_key: PRNGKey, state: BONGState, num_samples: int=10):
    """Sample from the diagonal-covariance Gaussian belief state"""
    mean, cov = state
    states = jr.normal(rng_key, shape=(num_samples, mean.shape[0])) \
        * jnp.sqrt(cov) + mean
    return states


class fg_bong:
    """Full-covariance Gaussian BONG algorithm.
    
    Parameters
    ----------
    init_mean : ArrayLikeTree
        Initial mean of the belief state.
    init_cov : ArrayLikeTree
        Initial covariance of the belief state.
    log_likelihood : Callable
        Log-likelihood function (mean, cov, y -> float).
    emission_mean_function : Callable
        Emission mean function (param, x -> ArrayLikeTree).
    emission_cov_function : Callable
        Emission covariance function (param, x -> ArrayLikeTree).
    dynamics_decay : float, optional
        Decay factor for the dynamics, by default 1.0
    process_noise : ArrayLikeTree, optional
        Process noise, by default 0.0
    num_samples : int, optional
        Number of samples to use for the update, by default 10
    linplugin : bool, optional
        Whether to use the linearized plugin method, by default False
    empirical_fisher: bool, optional
        Whether to use the empirical Fisher approximation to the Hessian matrix.
    
    Returns
    -------
    A RebayesAlgorithm.
    
    """
    sample = staticmethod(sample_fg_bong)
    
    def __new__(
        cls,
        init_mean: ArrayLikeTree,
        init_cov: float,
        log_likelihood: Callable,
        emission_mean_function: Callable,
        emission_cov_function: Callable,
        dynamics_decay: float=1.0,
        process_noise: ArrayLikeTree=0.0,
        num_samples: int=10,
        linplugin: bool=False,
        empirical_fisher: bool=False,
    ):
        init_cov = init_cov * jnp.eye(len(init_mean))
        if isinstance(process_noise, (int, float)):
            process_noise = jax.tree_map(lambda x: process_noise, init_cov)
        if linplugin:
            _update_fn = staticmethod(update_lfg_bong)
        else:
            _update_fn = staticmethod(update_fg_bong)
            
        def init_fn() -> BONGState:
            return staticmethod(init_bong)(init_mean, init_cov)
            
        def pred_fn(state: BONGState) -> BONGState:
            return staticmethod(predict_bong)(
                state, dynamics_decay, process_noise
            )
        
        def update_fn(
            rng_key: PRNGKey, 
            state: BONGState, 
            x: ArrayLikeTree, 
            y: ArrayLikeTree
        ) -> BONGState:
            return _update_fn(
                rng_key, state, x, y, log_likelihood, emission_mean_function, 
                emission_cov_function, num_samples, empirical_fisher
            )   
        
        return RebayesAlgorithm(init_fn, pred_fn, update_fn, cls.sample)
    

class dg_bong:
    """Diagonal-covariance Gaussian BONG algorithm.
    
    Parameters
    ----------
    init_mean : ArrayLikeTree
        Initial mean of the belief state.
    init_cov : ArrayLikeTree
        Initial variance of the belief state.
    log_likelihood : Callable
        Log-likelihood function (mean, cov, y -> float).
    emission_mean_function : Callable
        Emission mean function (param, x -> ArrayLikeTree).
    emission_cov_function : Callable
        Emission covariance function (param, x -> ArrayLikeTree).
    dynamics_decay : float, optional
        Decay factor for the dynamics, by default 1.0
    process_noise : ArrayLikeTree, optional
        Process noise, by default 0.0
    num_samples : int, optional
        Number of samples to use for the update, by default 10
    linplugin : bool, optional
        Whether to use the linearized plugin method, by default False
    
    Returns
    -------
    A RebayesAlgorithm.
    
    """
    sample = staticmethod(sample_dg_bong)
    
    def __new__(
        cls,
        init_mean: ArrayLikeTree,
        init_cov: ArrayLikeTree,
        log_likelihood: Callable,
        emission_mean_function: Callable,
        emission_cov_function: Callable,
        dynamics_decay: float=1.0,
        process_noise: ArrayLikeTree=0.0,
        num_samples: int=10,
        linplugin: bool=False,
    ):
        init_cov = init_cov * jnp.ones(len(init_mean))
        if isinstance(process_noise, (int, float)):
            process_noise = jax.tree_map(lambda x: process_noise, init_cov)
        if linplugin:
            _update_fn = staticmethod(update_ldg_bong)
        else:
            _update_fn = staticmethod(update_dg_bong)
            
        def init_fn() -> BONGState:
            return staticmethod(init_bong)(init_mean, init_cov)
            
        def pred_fn(state: BONGState) -> BONGState:
            return staticmethod(predict_bong)(
                state, dynamics_decay, process_noise
            )
        
        def update_fn(
            rng_key: PRNGKey, 
            state: BONGState, 
            x: ArrayLikeTree, 
            y: ArrayLikeTree
        ) -> BONGState:
            return _update_fn(
                rng_key, state, x, y, log_likelihood, emission_mean_function, 
                emission_cov_function, num_samples
            )   
        
        return RebayesAlgorithm(init_fn, pred_fn, update_fn, cls.sample)
    
    
class fg_reparam_bong:
    """Full-covariance Gaussian reparameterized BONG algorithm.
    
    Parameters
    ----------
    init_mean : ArrayLikeTree
        Initial mean of the belief state.
    init_cov : ArrayLikeTree
        Initial covariance of the belief state.
    log_likelihood : Callable
        Log-likelihood function (mean, cov, y -> float).
    emission_mean_function : Callable
        Emission mean function (param, x -> ArrayLikeTree).
    emission_cov_function : Callable
        Emission covariance function (param, x -> ArrayLikeTree).
    dynamics_decay : float, optional
        Decay factor for the dynamics, by default 1.0
    process_noise : ArrayLikeTree, optional
        Process noise, by default 0.0
    num_samples : int, optional
        Number of samples to use for the update, by default 10
    linplugin : bool, optional
        Whether to use the linearized plugin method, by default False
    empirical_fisher: bool, optional
        Whether to use the empirical Fisher approximation to the Hessian matrix.
    
    Returns
    -------
    A RebayesAlgorithm.
    
    """
    sample = staticmethod(sample_fg_bong)
    
    def __new__(
        cls,
        init_mean: ArrayLikeTree,
        init_cov: ArrayLikeTree,
        log_likelihood: Callable,
        emission_mean_function: Callable,
        emission_cov_function: Callable,
        dynamics_decay: float=1.0,
        process_noise: ArrayLikeTree=0.0,
        num_samples: int=10,
        linplugin: bool=False,
        empirical_fisher: bool=False,
    ):
        init_cov = init_cov * jnp.eye(len(init_mean))
        if isinstance(process_noise, (int, float)):
            process_noise = jax.tree_map(lambda x: process_noise, init_cov)
        if linplugin:
            _update_fn = staticmethod(update_lfg_reparam_bong)
        else:
            _update_fn = staticmethod(update_fg_reparam_bong)
            
        def init_fn() -> BONGState:
            return staticmethod(init_bong)(init_mean, init_cov)
            
        def pred_fn(state: BONGState) -> BONGState:
            return staticmethod(predict_bong)(
                state, dynamics_decay, process_noise
            )
        
        def update_fn(
            rng_key: PRNGKey, 
            state: BONGState, 
            x: ArrayLikeTree, 
            y: ArrayLikeTree
        ) -> BONGState:
            return _update_fn(
                rng_key, state, x, y, log_likelihood, emission_mean_function, 
                emission_cov_function, num_samples, empirical_fisher
            )   
        
        return RebayesAlgorithm(init_fn, pred_fn, update_fn, cls.sample)
    

class dg_reparam_bong:
    """Diagonal-covariance Gaussian reparameterized BONG algorithm.
    
    Parameters
    ----------
    init_mean : ArrayLikeTree
        Initial mean of the belief state.
    init_cov : ArrayLikeTree
        Initial variance of the belief state.
    log_likelihood : Callable
        Log-likelihood function (mean, cov, y -> float).
    emission_mean_function : Callable
        Emission mean function (param, x -> ArrayLikeTree).
    emission_cov_function : Callable
        Emission covariance function (param, x -> ArrayLikeTree).
    dynamics_decay : float, optional
        Decay factor for the dynamics, by default 1.0
    process_noise : ArrayLikeTree, optional
        Process noise, by default 0.0
    num_samples : int, optional
        Number of samples to use for the update, by default 10
    linplugin : bool, optional
        Whether to use the linearized plugin method, by default False
    
    Returns
    -------
    A RebayesAlgorithm.
    
    """
    sample = staticmethod(sample_dg_bong)
    
    def __new__(
        cls,
        init_mean: ArrayLikeTree,
        init_cov: ArrayLikeTree,
        log_likelihood: Callable,
        emission_mean_function: Callable,
        emission_cov_function: Callable,
        dynamics_decay: float=1.0,
        process_noise: ArrayLikeTree=0.0,
        num_samples: int=10,
        linplugin: bool=False,
    ):
        init_cov = init_cov * jnp.ones(len(init_mean))
        if isinstance(process_noise, (int, float)):
            process_noise = jax.tree_map(lambda x: process_noise, init_cov)
        if linplugin:
            _update_fn = staticmethod(update_ldg_reparam_bong)
        else:
            _update_fn = staticmethod(update_dg_reparam_bong)
            
        def init_fn() -> BONGState:
            return staticmethod(init_bong)(init_mean, init_cov)
            
        def pred_fn(state: BONGState) -> BONGState:
            return staticmethod(predict_bong)(
                state, dynamics_decay, process_noise
            )
        
        def update_fn(
            rng_key: PRNGKey, 
            state: BONGState, 
            x: ArrayLikeTree, 
            y: ArrayLikeTree
        ) -> BONGState:
            return _update_fn(
                rng_key, state, x, y, log_likelihood, emission_mean_function, 
                emission_cov_function, num_samples
            )   
        
        return RebayesAlgorithm(init_fn, pred_fn, update_fn, cls.sample)
    