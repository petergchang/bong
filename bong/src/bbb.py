from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr

from bong.base import RebayesAlgorithm
from bong.src.bong import sample_dg_bong, sample_fg_bong
from bong.types import ArrayLikeTree, ArrayTree, PRNGKey


class BBBState(NamedTuple):
    """Belief state of a BBB agent.
    
    mean: Mean of the belief state.
    cov: Covariance of the belief state.
    """
    mean: ArrayTree
    cov: ArrayTree
    

def init_bbb(
    init_mean: ArrayLikeTree,
    init_cov: ArrayLikeTree,
) -> BBBState:
    """Initialize the belief state with a mean and precision.
    
    Args:
        init_mean: Initial mean of the belief state.
        init_cov: Initial covariance of the belief state.
    
    Returns:
        Initial belief state.
    """
    return BBBState(mean=init_mean, cov=init_cov)


def predict_bbb(
    state: BBBState,
    gamma: float,
    Q: ArrayLikeTree,
    *args,
    **kwargs,
) -> BBBState:
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
    new_state = BBBState(new_mean, new_cov)
    return new_state


def update_fg_bbb(
    rng_key: PRNGKey,
    state_pred: BBBState,
    state: BBBState,
    x: ArrayLikeTree,
    y: ArrayLikeTree,
    log_likelihood: Callable,
    emission_mean_function: Callable,
    emission_cov_function: Callable,
    num_samples: int=10,
    empirical_fisher: bool=False,
    learning_rate: float=1.0,
    *args,
    **kwargs,
) -> BBBState:
    """Update the full-covariance Gaussian belief state with a new observation.
    
    Args:
        rng_key: JAX PRNG Key.
        state_pred: Belief state from the predict step.
        state: Current belief state.
        x: Input.
        y: Observation.
        log_likelihood: Log-likelihood function.
        emission_mean_function: Emission mean function.
        emission_cov_function: Emission covariance function.
        num_samples: Number of samples to use for the update.
        empirical_fisher: Whether to use the empirical Fisher approximation
            to the Hessian matrix.
        learning_rate: Learning rate for the update.
    
    Returns:
        Updated belief state.
    """
    mean0, cov0 = state_pred
    mean, cov = state
    
    def ll_fn(param):
        emission_mean = emission_mean_function(param, x)
        emission_cov = emission_cov_function(param, x)
        return jnp.mean(log_likelihood(emission_mean, emission_cov, y))
    
    z = sample_fg_bong(rng_key, state, num_samples)
    grads = jax.vmap(jax.grad(ll_fn))(z)
    if empirical_fisher:
        hess = -1/num_samples * grads.T @ grads
    else:
        hess = jnp.mean(jax.vmap(jax.hessian(ll_fn))(z), axis=0)
    g = jnp.mean(grads, axis=0)
    prec0, prec = jnp.linalg.pinv(cov0), jnp.linalg.pinv(cov)
    prec_update = 2 * jnp.outer(g, mean) \
        + 2 * prec0 @ jnp.outer(mean0 - mean, mean) + (hess - prec0) @ cov \
        + jnp.eye(cov.shape[0])
    new_prec = prec - 2 * learning_rate * cov @ prec_update
    new_cov = jnp.linalg.pinv(new_prec)
    mean_update = g + prec0 @ (mean0 - mean)
    new_mean = new_cov @ prec @ mean \
        + learning_rate * new_cov @ cov @ mean_update
    new_state = BBBState(new_mean, new_cov)
    return new_state


def update_dg_bbb(
    rng_key: PRNGKey,
    state_pred: BBBState,
    state: BBBState,
    x: ArrayLikeTree,
    y: ArrayLikeTree,
    log_likelihood: Callable,
    emission_mean_function: Callable,
    emission_cov_function: Callable,
    num_samples: int=10,
    empirical_fisher: bool=False,
    learning_rate: float=1.0,
    *args,
    **kwargs,
) -> BBBState:
    """Update the diagonal-covariance Gaussian belief state 
    with a new observation.
    
    Args:
        rng_key: JAX PRNG Key.
        state_pred: Belief state from the predict step.
        state: Current belief state.
        x: Input.
        y: Observation.
        log_likelihood: Log-likelihood function.
        emission_mean_function: Emission mean function.
        emission_cov_function: Emission covariance function.
        num_samples: Number of samples to use for the update.
        empirical_fisher: Whether to use the empirical Fisher approximation
            to the Hessian matrix.
        learning_rate: Learning rate for the update.
    
    Returns:
        Updated belief state.
    """
    mean0, cov0 = state_pred
    mean, cov = state
    
    def ll_fn(param):
        emission_mean = emission_mean_function(param, x)
        emission_cov = emission_cov_function(param, x)
        return jnp.mean(log_likelihood(emission_mean, emission_cov, y))
    
    keys = jr.split(rng_key)
    z = sample_dg_bong(keys[0], state, num_samples)
    grads = jax.vmap(jax.grad(ll_fn))(z)
    g = jnp.mean(grads, axis=0)
    if empirical_fisher:
        hess_diag = -1/num_samples * jnp.einsum('ij,ij->j', grads, grads)
    else:
        hess_diag_fn = lambda param: hess_diag_approx(keys[1], ll_fn, param)
        hess_diag = jnp.mean(jax.vmap(hess_diag_fn)(z), axis=0)
    prec0, prec = 1/cov0, 1/cov
    prec_update = 2 * g * mean + 2 * prec0 * (mean0-mean) * mean \
        + cov * (hess_diag - prec0) + 1
    new_prec = prec - 2 * learning_rate * cov * prec_update
    new_cov = 1 / new_prec
    mean_update = g + prec0 * (mean0 - mean)
    new_mean = new_cov * prec * mean \
        + learning_rate * new_cov * cov * mean_update
    new_state = BBBState(new_mean, new_cov)
    return new_state


def update_fg_reparam_bbb(
    rng_key: PRNGKey,
    state_pred: BBBState,
    state: BBBState,
    x: ArrayLikeTree,
    y: ArrayLikeTree,
    log_likelihood: Callable,
    emission_mean_function: Callable,
    emission_cov_function: Callable,
    num_samples: int=10,
    empirical_fisher: bool=False,
    learning_rate: float=1.0,
    *args,
    **kwargs,
) -> BBBState:
    """Update the full-covariance Gaussian belief state with a new observation,
    under the reparameterized BBB model.
    
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
        learning_rate: Learning rate for the update.
    
    Returns:
        Updated belief state.
    """
    mean0, cov0 = state_pred
    mean, cov = state
    
    def ll_fn(param):
        emission_mean = emission_mean_function(param, x)
        emission_cov = emission_cov_function(param, x)
        return jnp.mean(log_likelihood(emission_mean, emission_cov, y))
    
    z = sample_fg_bong(rng_key, state, num_samples)
    grads = jax.vmap(jax.grad(ll_fn))(z)
    if empirical_fisher:
        hess = -1/num_samples * grads.T @ grads
    else:
        hess = jnp.mean(jax.vmap(jax.hessian(ll_fn))(z), axis=0)
    g = jnp.mean(grads, axis=0)
    prec0, prec = jnp.linalg.pinv(cov0), jnp.linalg.pinv(cov)
    mean_update = g + prec0 @ (mean0 - mean)
    new_mean = mean + learning_rate * mean_update
    cov_update = hess + prec + prec0
    new_cov = cov + learning_rate / 2 * cov_update
    new_state = BBBState(new_mean, new_cov)
    return new_state


def update_dg_reparam_bbb(
    rng_key: PRNGKey,
    state_pred: BBBState,
    state: BBBState,
    x: ArrayLikeTree,
    y: ArrayLikeTree,
    log_likelihood: Callable,
    emission_mean_function: Callable,
    emission_cov_function: Callable,
    num_samples: int=10,
    empirical_fisher: bool=False,
    learning_rate: float=1.0,
    *args,
    **kwargs,
) -> BBBState:
    """Update the diagonal-covariance Gaussian belief state with a new observation
    under the reparameterized BBB model.
    
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
        learning_rate: Learning rate for the update.
    
    Returns:
        Updated belief state.
    """
    mean0, cov0 = state_pred
    mean, cov = state
    
    def ll_fn(param):
        emission_mean = emission_mean_function(param, x)
        emission_cov = emission_cov_function(param, x)
        return jnp.mean(log_likelihood(emission_mean, emission_cov, y))
    
    keys = jr.split(rng_key)
    z = sample_dg_bong(keys[0], state, num_samples)
    grads = jax.vmap(jax.grad(ll_fn))(z)
    g = jnp.mean(grads, axis=0)
    if empirical_fisher:
        hess_diag = -1/num_samples * jnp.einsum('ij,ij->j', grads, grads)
    else:
        hess_diag_fn = lambda param: hess_diag_approx(keys[1], ll_fn, param)
        hess_diag = jnp.mean(jax.vmap(hess_diag_fn)(z), axis=0)
    prec0, prec = 1 / cov0, 1 / cov
    mean_update = g + prec0 * (mean0 - mean)
    new_mean = mean + learning_rate * mean_update
    cov_update = hess_diag + prec + prec0
    new_cov = cov + learning_rate/2 * cov_update
    new_state = BBBState(new_mean, new_cov)
    return new_state


class fg_bbb:
    """Full-covariance Gaussian BBB algorithm.
    
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
    learning_rate: float, optional
        Learning rate for the update.
    num_iter: int, optional
        Number of iterations per time step.
    
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
        learning_rate: float=1.0,
        num_iter: int=10,
    ):
        init_cov = init_cov * jnp.eye(len(init_mean))
        if isinstance(process_noise, (int, float)):
            process_noise = jax.tree_map(lambda x: process_noise, init_cov)
        if linplugin:
            # _update_fn = staticmethod(update_lfg_bbb) TODO
            raise NotImplementedError
        else:
            _update_fn = staticmethod(update_fg_bbb)
            
        def init_fn() -> BBBState:
            return staticmethod(init_bbb)(init_mean, init_cov)
            
        def pred_fn(state: BBBState) -> BBBState:
            return staticmethod(predict_bbb)(
                state, dynamics_decay, process_noise
            )
        
        def update_fn(
            rng_key: PRNGKey, 
            state: BBBState,
            x: ArrayLikeTree, 
            y: ArrayLikeTree
        ) -> BBBState:
            @jax.jit
            def _step(curr_state, t):
                key = jr.fold_in(rng_key, t)
                new_state = _update_fn(
                    rng_key, state, curr_state, x, y, log_likelihood, 
                    emission_mean_function, emission_cov_function, num_samples, 
                    empirical_fisher, learning_rate
                )
                return new_state, new_state
            
            new_state, _ = jax.lax.scan(_step, state, jnp.arange(num_iter))
            return new_state
        
        return RebayesAlgorithm(init_fn, pred_fn, update_fn, cls.sample)
    

class dg_bbb:
    """Diagonal-covariance Gaussian BBB algorithm.
    
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
    empirical_fisher: bool, optional
        Whether to use the empirical Fisher approximation to the Hessian matrix.
    learning_rate: float, optional
        Learning rate for the update.
    num_iter: int, optional
        Number of iterations per time step.
    
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
        empirical_fisher: bool=False,
        learning_rate: float=1.0,
        num_iter: int=10,
    ):
        init_cov = init_cov * jnp.ones(len(init_mean))
        if isinstance(process_noise, (int, float)):
            process_noise = jax.tree_map(lambda x: process_noise, init_cov)
        if linplugin:
            # _update_fn = staticmethod(update_ldg_bbb) TODO
            raise NotImplementedError
        else:
            _update_fn = staticmethod(update_dg_bbb)
            
        def init_fn() -> BBBState:
            return staticmethod(init_bbb)(init_mean, init_cov)
            
        def pred_fn(state: BBBState) -> BBBState:
            return staticmethod(predict_bbb)(
                state, dynamics_decay, process_noise
            )
        
        def update_fn(
            rng_key: PRNGKey, 
            state: BBBState, 
            x: ArrayLikeTree, 
            y: ArrayLikeTree
        ) -> BBBState:
            @jax.jit
            def _step(curr_state, t):
                key = jr.fold_in(rng_key, t)
                new_state = _update_fn(
                    rng_key, state, curr_state, x, y, log_likelihood, 
                    emission_mean_function, emission_cov_function, num_samples, 
                    empirical_fisher, learning_rate
                )
                return new_state, new_state
            
            new_state, _ = jax.lax.scan(_step, state, jnp.arange(num_iter))
            return new_state
        
        return RebayesAlgorithm(init_fn, pred_fn, update_fn, cls.sample)
    

class fg_reparam_bbb:
    """Full-covariance Gaussian reparameterized BBB algorithm.
    
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
    learning_rate: float, optional
        Learning rate for the update.
    num_iter: int, optional
        Number of iterations per time step.
    
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
        learning_rate: float=1.0,
        num_iter: int=10,
    ):
        init_cov = init_cov * jnp.eye(len(init_mean))
        if isinstance(process_noise, (int, float)):
            process_noise = jax.tree_map(lambda x: process_noise, init_cov)
        if linplugin:
            # _update_fn = staticmethod(update_lfg_reparam_bbb) TODO
            raise NotImplementedError
        else:
            _update_fn = staticmethod(update_fg_reparam_bbb)
            
        def init_fn() -> BBBState:
            return staticmethod(init_bbb)(init_mean, init_cov)
            
        def pred_fn(state: BBBState) -> BBBState:
            return staticmethod(predict_bbb)(
                state, dynamics_decay, process_noise
            )
        
        def update_fn(
            rng_key: PRNGKey, 
            state: BBBState,
            x: ArrayLikeTree, 
            y: ArrayLikeTree
        ) -> BBBState:
            @jax.jit
            def _step(curr_state, t):
                key = jr.fold_in(rng_key, t)
                new_state = _update_fn(
                    rng_key, state, curr_state, x, y, log_likelihood, 
                    emission_mean_function, emission_cov_function, num_samples, 
                    empirical_fisher, learning_rate
                )
                return new_state, new_state
            
            new_state, _ = jax.lax.scan(_step, state, jnp.arange(num_iter))
            return new_state   
        
        return RebayesAlgorithm(init_fn, pred_fn, update_fn, cls.sample)
    

class dg_reparam_bbb:
    """Diagonal-covariance Gaussian reparameterized BBB algorithm.
    
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
    empirical_fisher: bool, optional
        Whether to use the empirical Fisher approximation to the Hessian matrix.
    learning_rate: float, optional
        Learning rate for the update.
    num_iter: int, optional
        Number of iterations per time step.
    
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
        empirical_fisher: bool=False,
        learning_rate: float=1.0,
        num_iter: int=10,
    ):
        init_cov = init_cov * jnp.ones(len(init_mean))
        if isinstance(process_noise, (int, float)):
            process_noise = jax.tree_map(lambda x: process_noise, init_cov)
        if linplugin:
            # _update_fn = staticmethod(update_ldg_reparam_bbb) TODO
            raise NotImplementedError
        else:
            _update_fn = staticmethod(update_dg_reparam_bbb)
            
        def init_fn() -> BBBState:
            return staticmethod(init_bbb)(init_mean, init_cov)
            
        def pred_fn(state: BBBState) -> BBBState:
            return staticmethod(predict_bbb)(
                state, dynamics_decay, process_noise
            )
        
        def update_fn(
            rng_key: PRNGKey, 
            state: BBBState,
            x: ArrayLikeTree, 
            y: ArrayLikeTree
        ) -> BBBState:
            @jax.jit
            def _step(curr_state, t):
                key = jr.fold_in(rng_key, t)
                new_state = _update_fn(
                    rng_key, state, curr_state, x, y, log_likelihood, 
                    emission_mean_function, emission_cov_function, num_samples, 
                    empirical_fisher, learning_rate
                )
                return new_state, new_state
            
            new_state, _ = jax.lax.scan(_step, state, jnp.arange(num_iter))
            return new_state
        
        return RebayesAlgorithm(init_fn, pred_fn, update_fn, cls.sample)
    