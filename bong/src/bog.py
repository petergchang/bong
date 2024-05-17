from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr

from bong.base import RebayesAlgorithm
from bong.src.bong import (
    sample_dg_bong,
    sample_dlrg_bong,
    sample_fg_bong
)
from bong.types import ArrayLikeTree, ArrayTree, PRNGKey
from bong.util import hess_diag_approx, safestr


class BOGState(NamedTuple):
    """Belief state of a BOG agent.
    
    mean: Mean of the belief state.
    cov: Covariance of the belief state.
    """
    mean: ArrayTree
    cov: ArrayTree
    

class BOGDLRState(NamedTuple):
    """Belief state of a DLR-BOG agent.
    
    mean: Mean of the belief state.
    prec_diag: Diagonal term (Upsilon) of DLR approximation of precision.
    prec_lr: Low-rank term (W) of DLR approximation of precision.
    """
    mean: ArrayTree
    prec_diag: ArrayTree
    prec_lr: ArrayTree
    

def init_bog(
    init_mean: ArrayLikeTree,
    init_cov: ArrayLikeTree,
) -> BOGState:
    """Initialize the belief state with a mean and precision.
    
    Args:
        init_mean: Initial mean of the belief state.
        init_cov: Initial covariance of the belief state.
    
    Returns:
        Initial belief state.
    """
    return BOGState(mean=init_mean, cov=init_cov)


def init_bog_dlr(
    init_mean: ArrayLikeTree,
    init_prec_diag: ArrayLikeTree,
    init_prec_lr: ArrayLikeTree,
) -> BOGDLRState:
    """Initialize the belief state with a mean and DLR precision.
    
    Args:
        init_mean: Initial mean of the belief state.
        init_prec_diag: Initial Upsilon of belief state.
        init_prec_lr: Initial W of belief state.
    
    Returns:
        Initial belief state.
    """
    return BOGDLRState(init_mean, init_prec_diag, init_prec_lr)


def predict_bog(
    state: BOGState,
    gamma: float,
    Q: ArrayLikeTree,
    *args,
    **kwargs,
) -> BOGState:
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
    new_state = BOGState(new_mean, new_cov)
    return new_state


def predict_bog_dlr(
    state: BOGDLRState,
    gamma: float,
    q: float,
    *args,
    **kwargs,
) -> BOGDLRState:
    """Predict the next state of the belief state.
    
    Args:
        state: Current belief state.
        gamma: Forgetting factor.
        Q: Process noise.
    
    Returns:
        Predicted belief state.
    """
    mean, prec_diag, prec_lr = state
    P, L = prec_lr.shape
    new_mean = jax.tree_map(lambda x: gamma * x, mean)
    new_prec_diag = 1/(gamma**2/prec_diag + q)
    C = jnp.linalg.pinv(
        jnp.eye(L) + q * prec_lr.T @ (prec_lr * (new_prec_diag/prec_diag))
    )
    new_prec_lr = \
        gamma * (new_prec_diag/prec_diag) * prec_lr @ jnp.linalg.cholesky(C)
    new_state = BOGDLRState(new_mean, new_prec_diag, new_prec_lr)
    return new_state


def update_fg_bog(
    rng_key: PRNGKey,
    state: BOGState,
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
) -> BOGState:
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
        learning_rate: Learning rate for the update.
    
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
        hess = -1/num_samples * grads.T @ grads
    else:
        hess = jnp.mean(jax.vmap(jax.hessian(ll_fn))(z), axis=0)
    g = jnp.mean(grads, axis=0)
    prec = jnp.linalg.pinv(cov)
    new_prec = prec - 4 * learning_rate * jnp.outer(cov @ g, mean) \
        - 2 * learning_rate * cov @ hess @ cov
    new_cov = jnp.linalg.pinv(new_prec)
    mean_update = prec @ mean + learning_rate * cov @ g
    new_mean = new_cov @ mean_update
    new_state = BOGState(new_mean, new_cov)
    return new_state


def update_lfg_bog(
    rng_key: PRNGKey,
    state: BOGState,
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
) -> BOGState:
    """Update the linearized-plugin full-covariance Gaussian belief state 
    with a new observation.
    
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
    mean, cov = state
    y_pred = jnp.atleast_1d(emission_mean_function(mean, x))
    H = jnp.atleast_2d(jax.jacrev(emission_mean_function)(mean, x))
    R = jnp.atleast_2d(emission_cov_function(mean, x))
    R_inv = jnp.linalg.pinv(R)
    prec = jnp.linalg.pinv(cov)
    update_term = cov @ H.T @ R_inv @ (y - y_pred)
    new_prec = prec - 4 * learning_rate * jnp.outer(update_term, mean) \
        + 2 * learning_rate * cov @ H.T @ R_inv @ H @ cov
    new_cov = jnp.linalg.pinv(new_prec)
    mean_update = prec @ mean + learning_rate * cov @ H.T @ R_inv @ (y - y_pred)
    new_mean = new_cov @ mean_update
    new_state = BOGState(new_mean, new_cov)
    return new_state


def update_dlrg_bog(
    rng_key: PRNGKey,
    state: BOGDLRState,
    x: ArrayLikeTree,
    y: ArrayLikeTree,
    log_likelihood: Callable,
    emission_mean_function: Callable,
    emission_cov_function: Callable,
    num_samples: int=10,
    empirical_fisher: bool=True,
    learning_rate: float=1.0,
    *args,
    **kwargs,
) -> BOGDLRState:
    """Update the DLR-precision Gaussian belief state with a new observation.
    
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
    if not empirical_fisher:
        raise NotImplementedError
    mean, prec_diag, prec_lr = state
    P, L = prec_lr.shape
    
    def ll_fn(param):
        emission_mean = emission_mean_function(param, x)
        emission_cov = emission_cov_function(param, x)
        return jnp.mean(log_likelihood(emission_mean, emission_cov, y))
    
    z = sample_dlrg_bong(rng_key, state, num_samples)
    grads = jax.vmap(jax.grad(ll_fn))(z)
    g = jnp.mean(grads, axis=0)
    prec_update = grads.T.reshape(P, -1)
    G = jnp.linalg.pinv(
        jnp.eye(L) + prec_lr.T @ (prec_lr / prec_diag)
    )
    B = prec_update / prec_diag - (prec_lr/prec_diag @ G) @ \
        ((prec_lr/prec_diag).T @ prec_update)
    new_mean = mean + learning_rate * g
    new_prec_diag = prec_diag + \
        learning_rate / (2*num_samples) * (B**2).sum(-1)[:, jnp.newaxis]
    new_prec_lr = prec_lr + learning_rate / num_samples * B @ (B.T @ prec_lr)
    new_state = BOGDLRState(new_mean, new_prec_diag, new_prec_lr)
    return new_state


def update_ldlrg_bog(
    rng_key: PRNGKey,
    state: BOGDLRState,
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
) -> BOGDLRState:
    """Update the linearized-plugin DLR-precision Gaussian belief state
    with a new observation.
    
    Args:
        rng_key: JAX PRNG Key.
        state: Current belief state.
        x: Input.
        y: Observation.
        log_likelihood: Log-likelihood function.
        emission_mean_function: Emission mean function.
        emission_cov_function: Emission covariance function.
        learning_rate: Learning rate for the update.
    
    Returns:
        Updated belief state.
    """
    num_samples = 1 # using linearized approximation, M=1
    mean, prec_diag, prec_lr = state
    P, L = prec_lr.shape
    y_pred = jnp.atleast_1d(emission_mean_function(mean, x))
    H = jnp.atleast_2d(jax.jacrev(emission_mean_function)(mean, x))
    R = jnp.atleast_2d(emission_cov_function(mean, x))
    R_chol = jnp.linalg.cholesky(R)
    A = jnp.linalg.lstsq(R_chol, jnp.eye(R.shape[0]))[0].T
    G = jnp.linalg.pinv(
        jnp.eye(L) + prec_lr.T @ (prec_lr / prec_diag)
    )
    prec_update = H.T @ A
    B = prec_update / prec_diag - (prec_lr/prec_diag @ G) @ \
        ((prec_lr/prec_diag).T @ prec_update)
    new_mean = mean + learning_rate * H.T @ A @ A.T @ (y - y_pred)
    new_prec_diag = prec_diag + \
        learning_rate / (2*num_samples) * (B**2).sum(-1)[:, jnp.newaxis]
    new_prec_lr = prec_lr + learning_rate / num_samples * B @ (B.T @ prec_lr)
    new_state = BOGDLRState(new_mean, new_prec_diag, new_prec_lr)
    return new_state


def update_dg_bog(
    rng_key: PRNGKey,
    state: BOGState,
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
) -> BOGState:
    """Update the diagonal-covariance Gaussian belief state 
    with a new observation.
    
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
    mean, cov = state
    
    def ll_fn(param):
        emission_mean = emission_mean_function(param, x)
        emission_cov = emission_cov_function(param, x)
        return jnp.mean(log_likelihood(emission_mean, emission_cov, y))
    
    keys = jr.split(rng_key)
    z = sample_dg_bong(keys[0], state, num_samples)
    grads = jax.vmap(jax.grad(ll_fn))(z)
    grad_est = jnp.mean(grads, axis=0)
    if empirical_fisher:
        hess_diag = -1/num_samples * jnp.einsum('ij,ij->j', grads, grads)
    else:
        hess_diag_fn = lambda param: hess_diag_approx(keys[1], ll_fn, param)
        hess_diag = jnp.mean(jax.vmap(hess_diag_fn)(z), axis=0)
    prec = 1 / cov
    prec_update = - 4 * learning_rate * cov * mean * grad_est \
        - 2 * learning_rate * cov**2 * hess_diag
    new_prec = prec + prec_update
    new_cov = 1 / new_prec
    new_mean = new_cov * prec * mean + learning_rate * new_cov * cov * grad_est
    new_state = BOGState(new_mean, new_cov)
    return new_state


def update_ldg_bog(
    rng_key: PRNGKey,
    state: BOGState,
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
) -> BOGState:
    """Update the linearized-plugin diagonal-covariance Gaussian belief state 
    with a new observation.
    
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
    mean, cov = state
    y_pred = jnp.atleast_1d(emission_mean_function(mean, x))
    H = jnp.atleast_2d(jax.jacrev(emission_mean_function)(mean, x))
    R = jnp.atleast_2d(emission_cov_function(mean, x))
    R_inv = jnp.linalg.pinv(R)
    prec = 1 / cov
    update_term = H.T @ R_inv @ (y - y_pred)
    prec_update = - 4 * learning_rate * cov * mean @ update_term \
        + 2 * learning_rate * cov**2 * ((H.T @ R_inv) * H.T).sum(-1)
    new_prec = prec + prec_update
    new_cov = 1 / new_prec
    new_mean = new_cov * prec * mean + \
        learning_rate * new_cov * cov * update_term
    new_state = BOGState(new_mean, new_cov)
    return new_state


def update_fg_reparam_bog(
    rng_key: PRNGKey,
    state: BOGState,
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
) -> BOGState:
    """Update the full-covariance Gaussian belief state with a new observation,
    under the reparameterized BOG model.
    
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
    mean, cov = state
    
    def ll_fn(param):
        emission_mean = emission_mean_function(param, x)
        emission_cov = emission_cov_function(param, x)
        return jnp.mean(log_likelihood(emission_mean, emission_cov, y))
    
    z = sample_fg_bong(rng_key, state, num_samples)
    grads = jax.vmap(jax.grad(ll_fn))(z)
    if empirical_fisher:
        cov_update = -1/num_samples * grads.T @ grads
    else:
        cov_update = jnp.mean(jax.vmap(jax.hessian(ll_fn))(z), axis=0)
    mean_update = jnp.mean(grads, axis=0)
    new_mean = mean + learning_rate * mean_update
    new_cov = cov + learning_rate / 2 * cov_update
    new_state = BOGState(new_mean, new_cov)
    return new_state


def update_lfg_reparam_bog(
    rng_key: PRNGKey,
    state: BOGState,
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
) -> BOGState:
    """Update the linearized-plugin full-covariance Gaussian belief state 
    with a new observation, under the reparameterized BOG model.
    
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
    mean, cov = state
    y_pred = jnp.atleast_1d(emission_mean_function(mean, x))
    H = jnp.atleast_2d(jax.jacrev(emission_mean_function)(mean, x))
    R = jnp.atleast_2d(emission_cov_function(mean, x))
    R_inv = jnp.linalg.pinv(R)
    prec = jnp.linalg.pinv(cov)
    mean_update = H.T @ R_inv @ (y - y_pred)
    new_mean = mean + learning_rate * mean_update
    cov_update = H.T @ R_inv @ H
    new_cov = cov - learning_rate / 2 * cov_update
    new_state = BOGState(new_mean, new_cov)
    return new_state


def update_dg_reparam_bog(
    rng_key: PRNGKey,
    state: BOGState,
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
) -> BOGState:
    """Update the diagonal-covariance Gaussian belief state with a new observation
    under the reparameterized BOG model.
    
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
    mean, cov = state
    
    def ll_fn(param):
        emission_mean = emission_mean_function(param, x)
        emission_cov = emission_cov_function(param, x)
        return jnp.mean(log_likelihood(emission_mean, emission_cov, y))
    
    keys = jr.split(rng_key)
    z = sample_dg_bong(keys[0], state, num_samples)
    grads = jax.vmap(jax.grad(ll_fn))(z)
    grad_est = jnp.mean(grads, axis=0)
    if empirical_fisher:
        hess_diag = -1/num_samples * jnp.einsum('ij,ij->j', grads, grads)
    else:
        hess_diag_fn = lambda param: hess_diag_approx(keys[1], ll_fn, param)
        hess_diag = jnp.mean(jax.vmap(hess_diag_fn)(z), axis=0)
    new_mean = mean + learning_rate * grad_est
    new_cov = cov + learning_rate/2 * hess_diag
    new_state = BOGState(new_mean, new_cov)
    return new_state


def update_ldg_reparam_bog(
    rng_key: PRNGKey,
    state: BOGState,
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
) -> BOGState:
    """Update the linearized-plugin diagonal-covariance Gaussian belief state 
    with a new observation under the reparameterized BOG model.
    
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
    mean, cov = state
    y_pred = jnp.atleast_1d(emission_mean_function(mean, x))
    H = jnp.atleast_2d(jax.jacrev(emission_mean_function)(mean, x))
    R = jnp.atleast_2d(emission_cov_function(mean, x))
    R_inv = jnp.linalg.pinv(R)
    new_mean = mean + learning_rate * H.T @ R_inv @ (y - y_pred)
    new_cov = cov - learning_rate/2 * ((H.T @ R_inv) * H.T).sum(-1)
    new_state = BOGState(new_mean, new_cov)
    return new_state


class fg_bog:
    """Full-covariance Gaussian BOG algorithm.
    
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
        learning_rate: float=1e-1,
        **kwargs,
    ):

        full_name = make_full_name("bog", "fc", rank=0, linplugin, empirical_fisher, num_samples, num_iter=1, learning_rate)
        if linplugin:
            name = f"bog-fc-lin-LR{safestr(learning_rate)}"
        else:
           name = f"bog-fc-EF{empirical_fisher}-MC{num_samples}-LR{safestr(learning_rate)}"

        init_cov = init_cov * jnp.eye(len(init_mean))
        if isinstance(process_noise, (int, float)):
            process_noise = jax.tree_map(lambda x: process_noise, init_cov)
        if linplugin:
            _update_fn = staticmethod(update_lfg_bog)
        else:
            _update_fn = staticmethod(update_fg_bog)
            
        def init_fn() -> BOGState:
            return staticmethod(init_bog)(init_mean, init_cov)
            
        def pred_fn(state: BOGState) -> BOGState:
            return staticmethod(predict_bog)(
                state, dynamics_decay, process_noise
            )
        
        def update_fn(
            rng_key: PRNGKey, 
            state: BOGState, 
            x: ArrayLikeTree, 
            y: ArrayLikeTree
        ) -> BOGState:
            return _update_fn(
                rng_key, state, x, y, log_likelihood, emission_mean_function, 
                emission_cov_function, num_samples, empirical_fisher,
                learning_rate
            )   
        
        return RebayesAlgorithm(init_fn, pred_fn, update_fn, cls.sample, name, full_name)
    

class dlrg_bog:
    """DLR-precision Gaussian BOG algorithm.
    
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
    rank: int, optional
        Rank of the low-rank approximation.
    
    Returns
    -------
    A RebayesAlgorithm.
    
    """
    sample = staticmethod(sample_dlrg_bong)

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
        learning_rate: float=1e-1,
        rank: int=10,
        **kwargs,
    ):

        full_name = make_full_name("bog", "dlr", rank, linplugin, empirical_fisher, num_samples, num_iter=1, learning_rate)
        if linplugin:
            name = f"bog-dlr{rank}-lin-LR{safestr(learning_rate)}"
        else:
           name = f"bog-dlr{rank}-EF{empirical_fisher}-MC{num_samples}-LR{safestr(learning_rate)}"

        init_prec_diag = 1/init_cov * jnp.ones((len(init_mean), 1)) # Diagonal term
        init_lr = jnp.zeros((len(init_mean), rank)) # Low-rank term
        if linplugin:
            _update_fn = staticmethod(update_ldlrg_bog)
        else:
            _update_fn = staticmethod(update_dlrg_bog)
            
        def init_fn() -> BOGDLRState:
            return staticmethod(init_bog_dlr)(
                init_mean, init_prec_diag, init_lr
            )
            
        def pred_fn(state: BOGDLRState) -> BOGDLRState:
            return staticmethod(predict_bog_dlr)(
                state, dynamics_decay, process_noise
            )
        
        def update_fn(
            rng_key: PRNGKey, 
            state: BOGDLRState, 
            x: ArrayLikeTree, 
            y: ArrayLikeTree
        ) -> BOGDLRState:
            return _update_fn(
                rng_key, state, x, y, log_likelihood, emission_mean_function, 
                emission_cov_function, num_samples, empirical_fisher,
                learning_rate
            )
        
        return RebayesAlgorithm(init_fn, pred_fn, update_fn, cls.sample, name, full_name)
    
    
class dg_bog:
    """Diagonal-covariance Gaussian BOG algorithm.
    
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
    
    Returns
    -------
    A RebayesAlgorithm.
    
    """
    sample = staticmethod(sample_dg_bong)
    
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
        learning_rate: float=1e-1,
        **kwargs,
    ):
        full_name = make_full_name("bog", "diag", rank=0, linplugin, empirical_fisher, num_samples, num_iter=1, learning_rate)
        if linplugin:
            name = f"bog-diag-lin-LR{safestr(learning_rate)}"
        else:
           name = f"bog-diag-EF{empirical_fisher}-MC{num_samples}-LR{safestr(learning_rate)}"


        init_cov = init_cov * jnp.ones(len(init_mean))
        if isinstance(process_noise, (int, float)):
            process_noise = jax.tree_map(lambda x: process_noise, init_cov)
        if linplugin:
            _update_fn = staticmethod(update_ldg_bog)
        else:
            _update_fn = staticmethod(update_dg_bog)
            
        def init_fn() -> BOGState:
            return staticmethod(init_bog)(init_mean, init_cov)
            
        def pred_fn(state: BOGState) -> BOGState:
            return staticmethod(predict_bog)(
                state, dynamics_decay, process_noise
            )
        
        def update_fn(
            rng_key: PRNGKey, 
            state: BOGState, 
            x: ArrayLikeTree, 
            y: ArrayLikeTree
        ) -> BOGState:
            return _update_fn(
                rng_key, state, x, y, log_likelihood, emission_mean_function, 
                emission_cov_function, num_samples, empirical_fisher,
                learning_rate
            )   
        
        return RebayesAlgorithm(init_fn, pred_fn, update_fn, cls.sample, name)
    
    
class fg_reparam_bog:
    """Full-covariance Gaussian reparameterized BOG algorithm.
    
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
        learning_rate: float=1e-1,
        **kwargs,
    ):
        full_name = make_full_name("bog", "fc_mom", rank=0, linplugin, empirical_fisher, num_samples, num_iter=1, learning_rate)
        if linplugin:
            name = f"bog-fc_mom-lin-LR{safestr(learning_rate)}"
        else:
           name = f"bog-fc_mom-EF{empirical_fisher}-MC{num_samples}-LR{safestr(learning_rate)}"

        init_cov = init_cov * jnp.eye(len(init_mean))
        if isinstance(process_noise, (int, float)):
            process_noise = jax.tree_map(lambda x: process_noise, init_cov)
        if linplugin:
            _update_fn = staticmethod(update_lfg_reparam_bog)
        else:
            _update_fn = staticmethod(update_fg_reparam_bog)
            
        def init_fn() -> BOGState:
            return staticmethod(init_bog)(init_mean, init_cov)
            
        def pred_fn(state: BOGState) -> BOGState:
            return staticmethod(predict_bog)(
                state, dynamics_decay, process_noise
            )
        
        def update_fn(
            rng_key: PRNGKey, 
            state: BOGState, 
            x: ArrayLikeTree, 
            y: ArrayLikeTree
        ) -> BOGState:
            return _update_fn(
                rng_key, state, x, y, log_likelihood, emission_mean_function, 
                emission_cov_function, num_samples, empirical_fisher,
                learning_rate
            )   
        
        return RebayesAlgorithm(init_fn, pred_fn, update_fn, cls.sample, name, full_name)
    

class dg_reparam_bog:
    """Diagonal-covariance Gaussian reparameterized BOG algorithm.
    
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
    
    Returns
    -------
    A RebayesAlgorithm.
    
    """
    sample = staticmethod(sample_dg_bong)
    
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
        learning_rate: float=1e-1,
        **kwargs,
    ):
        full_name = make_full_name("bog", "diag_mom", rank=0, linplugin, empirical_fisher, num_samples, num_iter=1, learning_rate)
        if linplugin:
            name = f"bog-diag_mom-lin-LR{safestr(learning_rate)}"
        else:
           name = f"bog-diag_mom-EF{empirical_fisher}-MC{num_samples}-LR{safestr(learning_rate)}"

        init_cov = init_cov * jnp.ones(len(init_mean))
        if isinstance(process_noise, (int, float)):
            process_noise = jax.tree_map(lambda x: process_noise, init_cov)
        if linplugin:
            _update_fn = staticmethod(update_ldg_reparam_bog)
        else:
            _update_fn = staticmethod(update_dg_reparam_bog)
            
        def init_fn() -> BOGState:
            return staticmethod(init_bog)(init_mean, init_cov)
            
        def pred_fn(state: BOGState) -> BOGState:
            return staticmethod(predict_bog)(
                state, dynamics_decay, process_noise
            )
        
        def update_fn(
            rng_key: PRNGKey, 
            state: BOGState, 
            x: ArrayLikeTree, 
            y: ArrayLikeTree
        ) -> BOGState:
            return _update_fn(
                rng_key, state, x, y, log_likelihood, emission_mean_function, 
                emission_cov_function, num_samples, empirical_fisher,
                learning_rate
            )   
        
        return RebayesAlgorithm(init_fn, pred_fn, update_fn, cls.sample, name, full_name)
    