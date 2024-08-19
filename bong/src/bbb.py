from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr

from bong.base import RebayesAlgorithm
from bong.src.bong import sample_dg_bong, sample_dlrg_bong, sample_fg_bong
from bong.src.states import AgentState, DLRAgentState
from bong.custom_types import ArrayLike, PRNGKey
from bong.util import hess_diag_approx, make_full_name


def init_bbb(
    init_mean: ArrayLike,
    init_cov: ArrayLike,
) -> AgentState:
    """Initialize the belief state with a mean and precision.

    Args:
        init_mean: Initial mean of the belief state.
        init_cov: Initial covariance of the belief state.

    Returns:
        Initial belief state.
    """
    return AgentState(mean=init_mean, cov=init_cov)


def init_bbb_dlr(
    init_mean: ArrayLike,
    init_prec_diag: ArrayLike,
    init_prec_lr: ArrayLike,
) -> DLRAgentState:
    """Initialize the belief state with a mean and DLR precision.

    Args:
        init_mean: Initial mean of the belief state.
        init_prec_diag: Initial Upsilon of belief state.
        init_prec_lr: Initial W of belief state.

    Returns:
        Initial belief state.
    """
    return DLRAgentState(init_mean, init_prec_diag, init_prec_lr)


def predict_bbb(
    state: AgentState,
    gamma: float,
    Q: ArrayLike,
    *args,
    **kwargs,
) -> AgentState:
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
    new_state = AgentState(new_mean, new_cov)
    return new_state


def predict_bbb_dlr(
    state: DLRAgentState,
    gamma: float,
    q: float,
    *args,
    **kwargs,
) -> DLRAgentState:
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
    new_prec_diag = 1 / (gamma**2 / prec_diag + q)
    C = jnp.linalg.pinv(
        jnp.eye(L) + q * prec_lr.T @ (prec_lr * (new_prec_diag / prec_diag))
    )
    new_prec_lr = gamma * (new_prec_diag / prec_diag) * prec_lr @ jnp.linalg.cholesky(C)
    new_state = DLRAgentState(new_mean, new_prec_diag, new_prec_lr)
    return new_state


def update_fg_bbb(
    rng_key: PRNGKey,
    state_pred: AgentState,
    state: AgentState,
    x: ArrayLike,
    y: ArrayLike,
    log_likelihood: Callable,
    emission_mean_function: Callable,
    emission_cov_function: Callable,
    num_samples: int = 10,
    empirical_fisher: bool = False,
    learning_rate: float = 1.0,
    *args,
    **kwargs,
) -> AgentState:
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
        hess = -1 / num_samples * grads.T @ grads
    else:
        hess = jnp.mean(jax.vmap(jax.hessian(ll_fn))(z), axis=0)
    g = jnp.mean(grads, axis=0)
    prec0, prec = jnp.linalg.pinv(cov0), jnp.linalg.pinv(cov)
    prec_update = (
        2 * jnp.outer(g, mean)
        + 2 * prec0 @ jnp.outer(mean0 - mean, mean)
        + (hess - prec0) @ cov
        + jnp.eye(cov.shape[0])
    )
    new_prec = prec - 2 * learning_rate * cov @ prec_update
    new_cov = jnp.linalg.pinv(new_prec)
    mean_update = g + prec0 @ (mean0 - mean)
    new_mean = new_cov @ prec @ mean + learning_rate * new_cov @ cov @ mean_update
    new_state = AgentState(new_mean, new_cov)
    return new_state


def update_lfg_bbb(
    rng_key: PRNGKey,
    state_pred: AgentState,
    state: AgentState,
    x: ArrayLike,
    y: ArrayLike,
    log_likelihood: Callable,
    emission_mean_function: Callable,
    emission_cov_function: Callable,
    num_samples: int = 10,
    empirical_fisher: bool = False,
    learning_rate: float = 1.0,
    *args,
    **kwargs,
) -> AgentState:
    """Update the linearized-plugin full-covariance Gaussian belief state
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
    y_pred = jnp.atleast_1d(emission_mean_function(mean, x))
    H = jnp.atleast_2d(jax.jacrev(emission_mean_function)(mean, x))
    R = jnp.atleast_2d(emission_cov_function(mean, x))
    R_inv = jnp.linalg.pinv(R)
    prec0, prec = jnp.linalg.pinv(cov0), jnp.linalg.pinv(cov)
    update_term = H.T @ R_inv @ (y - y_pred)
    prec_update = (
        2 * jnp.outer(update_term, mean)
        + 2 * prec0 @ jnp.outer(mean0 - mean, mean)
        - (H.T @ R_inv @ H + prec0) @ cov
        + jnp.eye(cov.shape[0])
    )
    new_prec = prec - 2 * learning_rate * cov @ prec_update
    new_cov = jnp.linalg.pinv(new_prec)
    mean_update = update_term + prec0 @ (mean0 - mean)
    new_mean = new_cov @ prec @ mean + learning_rate * new_cov @ cov @ mean_update
    new_state = AgentState(new_mean, new_cov)
    return new_state


def update_dlrg_bbb(
    rng_key: PRNGKey,
    state_pred: DLRAgentState,
    state: DLRAgentState,
    x: ArrayLike,
    y: ArrayLike,
    log_likelihood: Callable,
    emission_mean_function: Callable,
    emission_cov_function: Callable,
    num_samples: int = 10,
    empirical_fisher: bool = False,
    learning_rate: float = 1.0,
    *args,
    **kwargs,
) -> DLRAgentState:
    """Update the DLR-precision Gaussian belief state with a new observation.

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
    if not empirical_fisher:
        raise NotImplementedError
    mean0, prec_diag0, prec_lr0 = state_pred
    mean, prec_diag, prec_lr = state
    P, L = prec_lr.shape

    def ll_fn(param):
        emission_mean = emission_mean_function(param, x)
        emission_cov = emission_cov_function(param, x)
        return jnp.mean(log_likelihood(emission_mean, emission_cov, y))

    z = sample_dlrg_bong(rng_key, state, num_samples)
    grads = jax.vmap(jax.grad(ll_fn))(z)
    g = jnp.mean(grads, axis=0)
    mean_update = prec_diag0.ravel() * (mean0 - mean) + prec_lr @ (
        prec_lr0.T @ (mean0 - mean)
    )
    new_mean = mean + learning_rate * (g + mean_update.ravel())
    C = jnp.linalg.pinv(jnp.eye(L) + prec_lr.T @ (prec_lr / prec_diag))
    # D @ Ups_t_t-1 @ D
    diag_term11 = 1 / prec_diag * prec_diag0 * 1 / prec_diag
    diag_term12 = -jnp.einsum(
        "ij,ij->i", diag_term11 * (prec_lr @ C), prec_lr / prec_diag
    ).reshape(-1, 1)
    diag_term13 = -jnp.einsum(
        "ij,ij->i", (prec_lr / prec_diag) @ C, diag_term11 * prec_lr
    ).reshape(-1, 1)
    diag_term14 = jnp.einsum(
        "ij,ji->i",
        (prec_lr / prec_diag) @ C,
        ((diag_term11 * prec_lr).T @ prec_lr) @ C @ (prec_lr / prec_diag).T,
    ).reshape(-1, 1)
    diag_term1 = diag_term11 + diag_term12 + diag_term13 + diag_term14

    # - D @ Ups_t_i @ D
    diag_term21 = -1 / prec_diag
    diag_term22 = 2 * jnp.einsum(
        "ij,ij->i", (prec_lr / prec_diag) @ C, prec_lr / prec_diag
    ).reshape(-1, 1)
    diag_term23 = -jnp.einsum(
        "ij,ji->i",
        (prec_lr / prec_diag) @ C,
        (prec_lr.T @ (prec_lr / prec_diag)) @ C @ (prec_lr / prec_diag).T,
    ).reshape(-1, 1)
    diag_term2 = diag_term21 + diag_term22 + diag_term23

    # D @ (G @ G^T)/M @ D
    grad_term = grads.T / jnp.sqrt(num_samples)
    diag_term31 = jnp.einsum(
        "ij,ij->i", grad_term / prec_diag, grad_term / prec_diag
    ).reshape(-1, 1)
    diag_term32 = -jnp.einsum(
        "ij,ij->i",
        (grad_term / prec_diag) @ (grad_term.T @ ((prec_lr / prec_diag) @ C)),
        (prec_lr / prec_diag),
    ).reshape(-1, 1)
    diag_term33 = -jnp.einsum(
        "ij,ij->i",
        (prec_lr / prec_diag) @ C,
        (grad_term / prec_diag) @ (grad_term.T @ (prec_lr / prec_diag)),
    ).reshape(-1, 1)
    diag_term34 = jnp.einsum(
        "ij,ij->i",
        (prec_lr / prec_diag) @ C,
        (prec_lr / prec_diag)
        @ C
        @ (prec_lr.T @ (grad_term / prec_diag) @ (grad_term.T @ (prec_lr / prec_diag))),
    ).reshape(-1, 1)
    diag_term3 = diag_term31 + diag_term32 + diag_term33 + diag_term34

    # D @ W_t_t-1 @ W_t_t-1.T @ D
    diag_term41 = jnp.einsum(
        "ij,ij->i", prec_lr0 / prec_diag, prec_lr0 / prec_diag
    ).reshape(-1, 1)
    diag_term42 = -jnp.einsum(
        "ij,ij->i",
        (prec_lr0 / prec_diag) @ (prec_lr0.T @ ((prec_lr / prec_diag) @ C)),
        (prec_lr / prec_diag),
    ).reshape(-1, 1)
    diag_term43 = -jnp.einsum(
        "ij,ij->i",
        (prec_lr / prec_diag) @ C,
        (prec_lr0 / prec_diag) @ (prec_lr0.T @ (prec_lr / prec_diag)),
    ).reshape(-1, 1)
    diag_term44 = jnp.einsum(
        "ij,ij->i",
        (prec_lr / prec_diag) @ C,
        (prec_lr / prec_diag)
        @ C
        @ (prec_lr.T @ (prec_lr0 / prec_diag) @ (prec_lr0.T @ (prec_lr / prec_diag))),
    ).reshape(-1, 1)
    diag_term4 = diag_term41 + diag_term42 + diag_term43 + diag_term44

    # - D @ W_t_i @ W_t_i.T @ D
    diag_term51 = -jnp.einsum(
        "ij,ij->i", prec_lr / prec_diag, prec_lr / prec_diag
    ).reshape(-1, 1)
    diag_term52 = jnp.einsum(
        "ij,ij->i",
        (prec_lr / prec_diag) @ (prec_lr.T @ ((prec_lr / prec_diag) @ C)),
        (prec_lr / prec_diag),
    ).reshape(-1, 1)
    diag_term53 = jnp.einsum(
        "ij,ij->i",
        (prec_lr / prec_diag) @ C,
        (prec_lr / prec_diag) @ (prec_lr.T @ (prec_lr / prec_diag)),
    ).reshape(-1, 1)
    diag_term54 = -jnp.einsum(
        "ij,ij->i",
        (prec_lr / prec_diag) @ C,
        (prec_lr / prec_diag)
        @ C
        @ (prec_lr.T @ (prec_lr / prec_diag) @ (prec_lr.T @ (prec_lr / prec_diag))),
    ).reshape(-1, 1)
    diag_term5 = diag_term51 + diag_term52 + diag_term53 + diag_term54

    prec_diag_update = diag_term1 + diag_term2 + diag_term3 + diag_term4 + diag_term5
    new_prec_diag = prec_diag + learning_rate / 2 * prec_diag_update

    lr_term0 = (prec_lr / prec_diag) @ C
    lr_term1 = (diag_term11 * prec_lr) @ C - (prec_lr / prec_diag) @ (
        C @ (prec_lr.T @ (diag_term11 * prec_lr) @ C)
    )
    lr_term2 = -lr_term0 + lr_term0 @ ((prec_lr / prec_diag).T @ prec_lr @ C)
    lr_term3 = (grad_term / prec_diag) @ (
        grad_term.T @ ((prec_lr / prec_diag) @ C)
    ) - lr_term0 @ (
        ((prec_lr / prec_diag).T @ grad_term) @ grad_term.T @ (prec_lr / prec_diag) @ C
    )
    lr_term4 = (prec_lr0 / prec_diag) @ (
        prec_lr0.T @ ((prec_lr / prec_diag) @ C)
    ) - lr_term0 @ (
        (prec_lr / prec_diag).T
        @ (prec_lr0 @ (prec_lr0.T @ ((prec_lr / prec_diag) @ C)))
    )
    lr_term5 = (prec_lr / prec_diag) @ (
        prec_lr.T @ ((prec_lr / prec_diag) @ C)
    ) - lr_term0 @ (
        (prec_lr / prec_diag).T @ (prec_lr @ (prec_lr.T @ ((prec_lr / prec_diag) @ C)))
    )
    lr_term = lr_term1 + lr_term2 + lr_term3 + lr_term4 + lr_term5
    new_prec_lr = prec_lr + learning_rate * lr_term
    new_state = DLRAgentState(new_mean, new_prec_diag, new_prec_lr)
    return new_state


def update_ldlrg_bbb(
    rng_key: PRNGKey,
    state_pred: DLRAgentState,
    state: DLRAgentState,
    x: ArrayLike,
    y: ArrayLike,
    log_likelihood: Callable,
    emission_mean_function: Callable,
    emission_cov_function: Callable,
    num_samples: int = 10,
    empirical_fisher: bool = False,
    learning_rate: float = 1.0,
    *args,
    **kwargs,
) -> DLRAgentState:
    """Update the linearized-plugin DLR-precision Gaussian belief state
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
        learning_rate: Learning rate for the update.

    Returns:
        Updated belief state.
    """
    mean0, prec_diag0, prec_lr0 = state_pred
    mean, prec_diag, prec_lr = state
    P, L = prec_lr.shape
    y_pred = jnp.atleast_1d(emission_mean_function(mean, x))
    H = jnp.atleast_2d(jax.jacrev(emission_mean_function)(mean, x))
    R = jnp.atleast_2d(emission_cov_function(mean, x))
    R_chol = jnp.linalg.cholesky(R)
    A = jnp.linalg.lstsq(R_chol, jnp.eye(R.shape[0]))[0].T
    Rinv = jnp.linalg.pinv(R)
    g = H.T @ Rinv @ (y - y_pred)
    grad_term = H.T @ A
    mean_update = prec_diag0.ravel() * (mean0 - mean) + prec_lr @ (
        prec_lr0.T @ (mean0 - mean)
    )
    new_mean = mean + learning_rate * (g + mean_update.ravel())
    C = jnp.linalg.pinv(jnp.eye(L) + prec_lr.T @ (prec_lr / prec_diag))

    # D @ Ups_t_t-1 @ D
    diag_term11 = 1 / prec_diag * prec_diag0 * 1 / prec_diag
    diag_term12 = -jnp.einsum(
        "ij,ij->i", diag_term11 * (prec_lr @ C), prec_lr / prec_diag
    ).reshape(-1, 1)
    diag_term13 = -jnp.einsum(
        "ij,ij->i", (prec_lr / prec_diag) @ C, diag_term11 * prec_lr
    ).reshape(-1, 1)
    diag_term14 = jnp.einsum(
        "ij,ji->i",
        (prec_lr / prec_diag) @ C,
        ((diag_term11 * prec_lr).T @ prec_lr) @ C @ (prec_lr / prec_diag).T,
    ).reshape(-1, 1)
    diag_term1 = diag_term11 + diag_term12 + diag_term13 + diag_term14

    # - D @ Ups_t_i @ D
    diag_term21 = -1 / prec_diag
    diag_term22 = 2 * jnp.einsum(
        "ij,ij->i", (prec_lr / prec_diag) @ C, prec_lr / prec_diag
    ).reshape(-1, 1)
    diag_term23 = -jnp.einsum(
        "ij,ji->i",
        (prec_lr / prec_diag) @ C,
        (prec_lr.T @ (prec_lr / prec_diag)) @ C @ (prec_lr / prec_diag).T,
    ).reshape(-1, 1)
    diag_term2 = diag_term21 + diag_term22 + diag_term23

    # D @ (G @ G^T)/M @ D
    diag_term31 = jnp.einsum(
        "ij,ij->i", grad_term / prec_diag, grad_term / prec_diag
    ).reshape(-1, 1)
    diag_term32 = -jnp.einsum(
        "ij,ij->i",
        (grad_term / prec_diag) @ (grad_term.T @ ((prec_lr / prec_diag) @ C)),
        (prec_lr / prec_diag),
    ).reshape(-1, 1)
    diag_term33 = -jnp.einsum(
        "ij,ij->i",
        (prec_lr / prec_diag) @ C,
        (grad_term / prec_diag) @ (grad_term.T @ (prec_lr / prec_diag)),
    ).reshape(-1, 1)
    diag_term34 = jnp.einsum(
        "ij,ij->i",
        (prec_lr / prec_diag) @ C,
        (prec_lr / prec_diag)
        @ C
        @ (prec_lr.T @ (grad_term / prec_diag) @ (grad_term.T @ (prec_lr / prec_diag))),
    ).reshape(-1, 1)
    diag_term3 = diag_term31 + diag_term32 + diag_term33 + diag_term34

    # D @ W_t_t-1 @ W_t_t-1.T @ D
    diag_term41 = jnp.einsum(
        "ij,ij->i", prec_lr0 / prec_diag, prec_lr0 / prec_diag
    ).reshape(-1, 1)
    diag_term42 = -jnp.einsum(
        "ij,ij->i",
        (prec_lr0 / prec_diag) @ (prec_lr0.T @ ((prec_lr / prec_diag) @ C)),
        (prec_lr / prec_diag),
    ).reshape(-1, 1)
    diag_term43 = -jnp.einsum(
        "ij,ij->i",
        (prec_lr / prec_diag) @ C,
        (prec_lr0 / prec_diag) @ (prec_lr0.T @ (prec_lr / prec_diag)),
    ).reshape(-1, 1)
    diag_term44 = jnp.einsum(
        "ij,ij->i",
        (prec_lr / prec_diag) @ C,
        (prec_lr / prec_diag)
        @ C
        @ (prec_lr.T @ (prec_lr0 / prec_diag) @ (prec_lr0.T @ (prec_lr / prec_diag))),
    ).reshape(-1, 1)
    diag_term4 = diag_term41 + diag_term42 + diag_term43 + diag_term44

    # - D @ W_t_i @ W_t_i.T @ D
    diag_term51 = -jnp.einsum(
        "ij,ij->i", prec_lr / prec_diag, prec_lr / prec_diag
    ).reshape(-1, 1)
    diag_term52 = jnp.einsum(
        "ij,ij->i",
        (prec_lr / prec_diag) @ (prec_lr.T @ ((prec_lr / prec_diag) @ C)),
        (prec_lr / prec_diag),
    ).reshape(-1, 1)
    diag_term53 = jnp.einsum(
        "ij,ij->i",
        (prec_lr / prec_diag) @ C,
        (prec_lr / prec_diag) @ (prec_lr.T @ (prec_lr / prec_diag)),
    ).reshape(-1, 1)
    diag_term54 = -jnp.einsum(
        "ij,ij->i",
        (prec_lr / prec_diag) @ C,
        (prec_lr / prec_diag)
        @ C
        @ (prec_lr.T @ (prec_lr / prec_diag) @ (prec_lr.T @ (prec_lr / prec_diag))),
    ).reshape(-1, 1)
    diag_term5 = diag_term51 + diag_term52 + diag_term53 + diag_term54

    prec_diag_update = diag_term1 + diag_term2 + diag_term3 + diag_term4 + diag_term5
    new_prec_diag = prec_diag + learning_rate / 2 * prec_diag_update

    lr_term0 = (prec_lr / prec_diag) @ C
    lr_term1 = (diag_term11 * prec_lr) @ C - (prec_lr / prec_diag) @ (
        C @ (prec_lr.T @ (diag_term11 * prec_lr) @ C)
    )
    lr_term2 = -lr_term0 + lr_term0 @ ((prec_lr / prec_diag).T @ prec_lr @ C)
    lr_term3 = (grad_term / prec_diag) @ (
        grad_term.T @ ((prec_lr / prec_diag) @ C)
    ) - lr_term0 @ (
        ((prec_lr / prec_diag).T @ grad_term) @ grad_term.T @ (prec_lr / prec_diag) @ C
    )
    lr_term4 = (prec_lr0 / prec_diag) @ (
        prec_lr0.T @ ((prec_lr / prec_diag) @ C)
    ) - lr_term0 @ (
        (prec_lr / prec_diag).T
        @ (prec_lr0 @ (prec_lr0.T @ ((prec_lr / prec_diag) @ C)))
    )
    lr_term5 = (prec_lr / prec_diag) @ (
        prec_lr.T @ ((prec_lr / prec_diag) @ C)
    ) - lr_term0 @ (
        (prec_lr / prec_diag).T @ (prec_lr @ (prec_lr.T @ ((prec_lr / prec_diag) @ C)))
    )
    lr_term = lr_term1 + lr_term2 + lr_term3 + lr_term4 + lr_term5
    new_prec_lr = prec_lr + learning_rate * lr_term
    new_state = DLRAgentState(new_mean, new_prec_diag, new_prec_lr)
    return new_state


def update_dg_bbb(
    rng_key: PRNGKey,
    state_pred: AgentState,
    state: AgentState,
    x: ArrayLike,
    y: ArrayLike,
    log_likelihood: Callable,
    emission_mean_function: Callable,
    emission_cov_function: Callable,
    num_samples: int = 10,
    empirical_fisher: bool = False,
    learning_rate: float = 1.0,
    *args,
    **kwargs,
) -> AgentState:
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
        hess_diag = -1 / num_samples * jnp.einsum("ij,ij->j", grads, grads)
    else:

        def hess_diag_fn(param):
            return hess_diag_approx(keys[1], ll_fn, param)

        hess_diag = jnp.mean(jax.vmap(hess_diag_fn)(z), axis=0)
    prec0, prec = 1 / cov0, 1 / cov
    prec_update = (
        2 * g * mean + 2 * prec0 * (mean0 - mean) * mean + cov * (hess_diag - prec0) + 1
    )
    new_prec = prec - 2 * learning_rate * cov * prec_update
    new_cov = 1 / new_prec
    mean_update = g + prec0 * (mean0 - mean)
    new_mean = new_cov * prec * mean + learning_rate * new_cov * cov * mean_update
    new_state = AgentState(new_mean, new_cov)
    return new_state


def update_ldg_bbb(
    rng_key: PRNGKey,
    state_pred: AgentState,
    state: AgentState,
    x: ArrayLike,
    y: ArrayLike,
    log_likelihood: Callable,
    emission_mean_function: Callable,
    emission_cov_function: Callable,
    num_samples: int = 10,
    empirical_fisher: bool = False,
    learning_rate: float = 1.0,
    *args,
    **kwargs,
) -> AgentState:
    """Update the linearized-plugin diagonal-covariance Gaussian belief state
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
    y_pred = jnp.atleast_1d(emission_mean_function(mean, x))
    H = jnp.atleast_2d(jax.jacrev(emission_mean_function)(mean, x))
    R = jnp.atleast_2d(emission_cov_function(mean, x))
    R_inv = jnp.linalg.pinv(R)
    prec0, prec = 1 / cov0, 1 / cov
    update_term = H.T @ R_inv @ (y - y_pred)
    prec_update = (
        2 * update_term * mean
        + 2 * prec0 * (mean0 - mean) * mean
        - cov * (((H.T @ R_inv) * H.T).sum(-1) + prec0)
        + 1
    )
    new_prec = prec - 2 * learning_rate * cov * prec_update
    new_cov = 1 / new_prec
    mean_update = update_term + prec0 * (mean0 - mean)
    new_mean = new_cov * prec * mean + learning_rate * new_cov * cov * mean_update
    new_state = AgentState(new_mean, new_cov)
    return new_state


def update_fg_reparam_bbb(
    rng_key: PRNGKey,
    state_pred: AgentState,
    state: AgentState,
    x: ArrayLike,
    y: ArrayLike,
    log_likelihood: Callable,
    emission_mean_function: Callable,
    emission_cov_function: Callable,
    num_samples: int = 10,
    empirical_fisher: bool = False,
    learning_rate: float = 1.0,
    *args,
    **kwargs,
) -> AgentState:
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
        hess = -1 / num_samples * grads.T @ grads
    else:
        hess = jnp.mean(jax.vmap(jax.hessian(ll_fn))(z), axis=0)
    g = jnp.mean(grads, axis=0)
    prec0, prec = jnp.linalg.pinv(cov0), jnp.linalg.pinv(cov)
    mean_update = g + prec0 @ (mean0 - mean)
    new_mean = mean + learning_rate * mean_update
    cov_update = hess + prec - prec0
    new_cov = cov + learning_rate / 2 * cov_update
    new_state = AgentState(new_mean, new_cov)
    return new_state


def update_lfg_reparam_bbb(
    rng_key: PRNGKey,
    state_pred: AgentState,
    state: AgentState,
    x: ArrayLike,
    y: ArrayLike,
    log_likelihood: Callable,
    emission_mean_function: Callable,
    emission_cov_function: Callable,
    num_samples: int = 10,
    empirical_fisher: bool = False,
    learning_rate: float = 1.0,
    *args,
    **kwargs,
) -> AgentState:
    """Update the linearized-plugin full-covariance Gaussian belief state
    with a new observation, under the reparameterized BBB model.

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
    y_pred = jnp.atleast_1d(emission_mean_function(mean, x))
    H = jnp.atleast_2d(jax.jacrev(emission_mean_function)(mean, x))
    R = jnp.atleast_2d(emission_cov_function(mean, x))
    R_inv = jnp.linalg.pinv(R)
    prec0, prec = jnp.linalg.pinv(cov0), jnp.linalg.pinv(cov)
    update_term = H.T @ R_inv @ (y - y_pred)
    mean_update = update_term + prec0 @ (mean0 - mean)
    new_mean = mean + learning_rate * mean_update
    cov_update = -H.T @ R_inv @ H + prec - prec0
    new_cov = cov + learning_rate / 2 * cov_update
    new_state = AgentState(new_mean, new_cov)
    return new_state


def update_dg_reparam_bbb(
    rng_key: PRNGKey,
    state_pred: AgentState,
    state: AgentState,
    x: ArrayLike,
    y: ArrayLike,
    log_likelihood: Callable,
    emission_mean_function: Callable,
    emission_cov_function: Callable,
    num_samples: int = 10,
    empirical_fisher: bool = False,
    learning_rate: float = 1.0,
    *args,
    **kwargs,
) -> AgentState:
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
        hess_diag = -1 / num_samples * jnp.einsum("ij,ij->j", grads, grads)
    else:

        def hess_diag_fn(param):
            return hess_diag_approx(keys[1], ll_fn, param)

        hess_diag = jnp.mean(jax.vmap(hess_diag_fn)(z), axis=0)
    prec0, prec = 1 / cov0, 1 / cov
    mean_update = g + prec0 * (mean0 - mean)
    new_mean = mean + learning_rate * mean_update
    cov_update = hess_diag + prec - prec0
    new_cov = cov + learning_rate / 2 * cov_update
    new_state = AgentState(new_mean, new_cov)
    return new_state


def update_ldg_reparam_bbb(
    rng_key: PRNGKey,
    state_pred: AgentState,
    state: AgentState,
    x: ArrayLike,
    y: ArrayLike,
    log_likelihood: Callable,
    emission_mean_function: Callable,
    emission_cov_function: Callable,
    num_samples: int = 10,
    empirical_fisher: bool = False,
    learning_rate: float = 1.0,
    *args,
    **kwargs,
) -> AgentState:
    """Update the linearized-plugin diagonal-covariance Gaussian belief state
    with a new observation under the reparameterized BBB model.

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
    y_pred = jnp.atleast_1d(emission_mean_function(mean, x))
    H = jnp.atleast_2d(jax.jacrev(emission_mean_function)(mean, x))
    R = jnp.atleast_2d(emission_cov_function(mean, x))
    R_inv = jnp.linalg.pinv(R)
    prec0, prec = 1 / cov0, 1 / cov
    update_term = H.T @ R_inv @ (y - y_pred)
    mean_update = update_term + prec0 * (mean0 - mean)
    new_mean = mean + learning_rate * mean_update
    cov_update = -((H.T @ R_inv) * H.T).sum(-1) + prec - prec0
    new_cov = cov + learning_rate / 2 * cov_update
    new_state = AgentState(new_mean, new_cov)
    return new_state


class fg_bbb:
    """Full-covariance Gaussian BBB algorithm.

    Parameters
    ----------
    init_mean : ArrayLike
        Initial mean of the belief state.
    init_cov : ArrayLike
        Initial covariance of the belief state.
    log_likelihood : Callable
        Log-likelihood function (mean, cov, y -> float).
    emission_mean_function : Callable
        Emission mean function (param, x -> ArrayLike).
    emission_cov_function : Callable
        Emission covariance function (param, x -> ArrayLike).
    dynamics_decay : float, optional
        Decay factor for the dynamics, by default 1.0
    process_noise : ArrayLike, optional
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
        init_mean: ArrayLike,
        init_cov: float,
        log_likelihood: Callable,
        emission_mean_function: Callable,
        emission_cov_function: Callable,
        dynamics_decay: float = 1.0,
        process_noise: ArrayLike = 0.0,
        num_samples: int = 10,
        linplugin: bool = False,
        empirical_fisher: bool = False,
        learning_rate: float = 1e-1,
        num_iter: int = 10,
        **kwargs,
    ):
        rank = 0
        full_name = make_full_name(
            "bbb",
            "fc",
            rank,
            linplugin,
            empirical_fisher,
            num_samples,
            num_iter,
            learning_rate,
        )
        name = full_name
        init_cov = init_cov * jnp.eye(len(init_mean))
        if isinstance(process_noise, (int, float)):
            process_noise = jax.tree_map(lambda x: process_noise, init_cov)
        if linplugin:
            _update_fn = staticmethod(update_lfg_bbb)
        else:
            _update_fn = staticmethod(update_fg_bbb)

        def init_fn() -> AgentState:
            return staticmethod(init_bbb)(init_mean, init_cov)

        def pred_fn(state: AgentState) -> AgentState:
            return staticmethod(predict_bbb)(state, dynamics_decay, process_noise)

        def update_fn(
            rng_key: PRNGKey, state: AgentState, x: ArrayLike, y: ArrayLike
        ) -> AgentState:
            @jax.jit
            def _step(curr_state, t):
                key = jr.fold_in(rng_key, t)
                new_state = _update_fn(
                    key,
                    state,
                    curr_state,
                    x,
                    y,
                    log_likelihood,
                    emission_mean_function,
                    emission_cov_function,
                    num_samples,
                    empirical_fisher,
                    learning_rate,
                )
                return new_state, new_state

            new_state, _ = jax.lax.scan(_step, state, jnp.arange(num_iter))
            return new_state

        return RebayesAlgorithm(
            init_fn, pred_fn, update_fn, cls.sample, name, full_name
        )


class dlrg_bbb:
    """DLR-precision Gaussian BBB algorithm.

    Parameters
    ----------
    init_mean : ArrayLike
        Initial mean of the belief state.
    init_cov : ArrayLike
        Initial covariance of the belief state.
    log_likelihood : Callable
        Log-likelihood function (mean, cov, y -> float).
    emission_mean_function : Callable
        Emission mean function (param, x -> ArrayLike).
    emission_cov_function : Callable
        Emission covariance function (param, x -> ArrayLike).
    dynamics_decay : float, optional
        Decay factor for the dynamics, by default 1.0
    process_noise : ArrayLike, optional
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
    rank: int, optional
        Rank of the low-rank approximation.

    Returns
    -------
    A RebayesAlgorithm.

    """

    sample = staticmethod(sample_dlrg_bong)

    def __new__(
        cls,
        init_mean: ArrayLike,
        init_cov: float,
        log_likelihood: Callable,
        emission_mean_function: Callable,
        emission_cov_function: Callable,
        dynamics_decay: float = 1.0,
        process_noise: ArrayLike = 0.0,
        num_samples: int = 10,
        linplugin: bool = False,
        empirical_fisher: bool = False,
        learning_rate: float = 1e-1,
        num_iter: int = 10,
        rank: int = 10,
        **kwargs,
    ):
        full_name = make_full_name(
            "bbb",
            "dlr",
            rank,
            linplugin,
            empirical_fisher,
            num_samples,
            num_iter,
            learning_rate,
        )
        name = full_name

        init_prec_diag = 1 / init_cov * jnp.ones((len(init_mean), 1))  # Diagonal term
        init_lr = jnp.zeros((len(init_mean), rank))  # Low-rank term
        if linplugin:
            _update_fn = staticmethod(update_ldlrg_bbb)
        else:
            _update_fn = staticmethod(update_dlrg_bbb)

        def init_fn() -> DLRAgentState:
            return staticmethod(init_bbb_dlr)(init_mean, init_prec_diag, init_lr)

        def pred_fn(state: DLRAgentState) -> DLRAgentState:
            return staticmethod(predict_bbb_dlr)(state, dynamics_decay, process_noise)

        def update_fn(
            rng_key: PRNGKey, state: DLRAgentState, x: ArrayLike, y: ArrayLike
        ) -> DLRAgentState:
            @jax.jit
            def _step(curr_state, t):
                key = jr.fold_in(rng_key, t)
                new_state = _update_fn(
                    key,
                    state,
                    curr_state,
                    x,
                    y,
                    log_likelihood,
                    emission_mean_function,
                    emission_cov_function,
                    num_samples,
                    empirical_fisher,
                    learning_rate,
                )
                return new_state, new_state

            new_state, _ = jax.lax.scan(_step, state, jnp.arange(num_iter))
            return new_state

        return RebayesAlgorithm(
            init_fn, pred_fn, update_fn, cls.sample, name, full_name
        )


class dg_bbb:
    """Diagonal-covariance Gaussian BBB algorithm.

    Parameters
    ----------
    init_mean : ArrayLike
        Initial mean of the belief state.
    init_cov : ArrayLike
        Initial variance of the belief state.
    log_likelihood : Callable
        Log-likelihood function (mean, cov, y -> float).
    emission_mean_function : Callable
        Emission mean function (param, x -> ArrayLike).
    emission_cov_function : Callable
        Emission covariance function (param, x -> ArrayLike).
    dynamics_decay : float, optional
        Decay factor for the dynamics, by default 1.0
    process_noise : ArrayLike, optional
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
        init_mean: ArrayLike,
        init_cov: float,
        log_likelihood: Callable,
        emission_mean_function: Callable,
        emission_cov_function: Callable,
        dynamics_decay: float = 1.0,
        process_noise: ArrayLike = 0.0,
        num_samples: int = 10,
        linplugin: bool = False,
        empirical_fisher: bool = False,
        learning_rate: float = 1e-1,
        num_iter: int = 10,
        **kwargs,
    ):
        rank = 99
        full_name = make_full_name(
            "bbb",
            "diag",
            rank,
            linplugin,
            empirical_fisher,
            num_samples,
            num_iter,
            learning_rate,
        )
        name = full_name

        init_cov = init_cov * jnp.ones(len(init_mean))
        if isinstance(process_noise, (int, float)):
            process_noise = jax.tree_map(lambda x: process_noise, init_cov)
        if linplugin:
            _update_fn = staticmethod(update_ldg_bbb)
        else:
            _update_fn = staticmethod(update_dg_bbb)

        def init_fn() -> AgentState:
            return staticmethod(init_bbb)(init_mean, init_cov)

        def pred_fn(state: AgentState) -> AgentState:
            return staticmethod(predict_bbb)(state, dynamics_decay, process_noise)

        def update_fn(
            rng_key: PRNGKey, state: AgentState, x: ArrayLike, y: ArrayLike
        ) -> AgentState:
            @jax.jit
            def _step(curr_state, t):
                key = jr.fold_in(rng_key, t)
                new_state = _update_fn(
                    key,
                    state,
                    curr_state,
                    x,
                    y,
                    log_likelihood,
                    emission_mean_function,
                    emission_cov_function,
                    num_samples,
                    empirical_fisher,
                    learning_rate,
                )
                return new_state, new_state

            new_state, _ = jax.lax.scan(_step, state, jnp.arange(num_iter))
            return new_state

        return RebayesAlgorithm(
            init_fn, pred_fn, update_fn, cls.sample, name, full_name
        )


class fg_reparam_bbb:
    """Full-covariance Gaussian reparameterized BBB algorithm.

    Parameters
    ----------
    init_mean : ArrayLike
        Initial mean of the belief state.
    init_cov : ArrayLike
        Initial covariance of the belief state.
    log_likelihood : Callable
        Log-likelihood function (mean, cov, y -> float).
    emission_mean_function : Callable
        Emission mean function (param, x -> ArrayLike).
    emission_cov_function : Callable
        Emission covariance function (param, x -> ArrayLike).
    dynamics_decay : float, optional
        Decay factor for the dynamics, by default 1.0
    process_noise : ArrayLike, optional
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
        init_mean: ArrayLike,
        init_cov: float,
        log_likelihood: Callable,
        emission_mean_function: Callable,
        emission_cov_function: Callable,
        dynamics_decay: float = 1.0,
        process_noise: ArrayLike = 0.0,
        num_samples: int = 10,
        linplugin: bool = False,
        empirical_fisher: bool = False,
        learning_rate: float = 1e-1,
        num_iter: int = 10,
        **kwargs,
    ):
        rank = 99
        full_name = make_full_name(
            "bbb",
            "fc_mom",
            rank,
            linplugin,
            empirical_fisher,
            num_samples,
            num_iter,
            learning_rate,
        )
        name = full_name

        init_cov = init_cov * jnp.eye(len(init_mean))
        if isinstance(process_noise, (int, float)):
            process_noise = jax.tree_map(lambda x: process_noise, init_cov)
        if linplugin:
            _update_fn = staticmethod(update_lfg_reparam_bbb)
        else:
            _update_fn = staticmethod(update_fg_reparam_bbb)

        def init_fn() -> AgentState:
            return staticmethod(init_bbb)(init_mean, init_cov)

        def pred_fn(state: AgentState) -> AgentState:
            return staticmethod(predict_bbb)(state, dynamics_decay, process_noise)

        def update_fn(
            rng_key: PRNGKey, state: AgentState, x: ArrayLike, y: ArrayLike
        ) -> AgentState:
            @jax.jit
            def _step(curr_state, t):
                key = jr.fold_in(rng_key, t)
                new_state = _update_fn(
                    key,
                    state,
                    curr_state,
                    x,
                    y,
                    log_likelihood,
                    emission_mean_function,
                    emission_cov_function,
                    num_samples,
                    empirical_fisher,
                    learning_rate,
                )
                return new_state, new_state

            new_state, _ = jax.lax.scan(_step, state, jnp.arange(num_iter))
            return new_state

        return RebayesAlgorithm(
            init_fn, pred_fn, update_fn, cls.sample, name, full_name
        )


class dg_reparam_bbb:
    """Diagonal-covariance Gaussian reparameterized BBB algorithm.

    Parameters
    ----------
    init_mean : ArrayLike
        Initial mean of the belief state.
    init_cov : ArrayLike
        Initial variance of the belief state.
    log_likelihood : Callable
        Log-likelihood function (mean, cov, y -> float).
    emission_mean_function : Callable
        Emission mean function (param, x -> ArrayLike).
    emission_cov_function : Callable
        Emission covariance function (param, x -> ArrayLike).
    dynamics_decay : float, optional
        Decay factor for the dynamics, by default 1.0
    process_noise : ArrayLike, optional
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
        init_mean: ArrayLike,
        init_cov: float,
        log_likelihood: Callable,
        emission_mean_function: Callable,
        emission_cov_function: Callable,
        dynamics_decay: float = 1.0,
        process_noise: ArrayLike = 0.0,
        num_samples: int = 10,
        linplugin: bool = False,
        empirical_fisher: bool = False,
        learning_rate: float = 1e-1,
        num_iter: int = 10,
        **kwargs,
    ):
        rank = 99
        full_name = make_full_name(
            "bbb",
            "diag_mom",
            rank,
            linplugin,
            empirical_fisher,
            num_samples,
            num_iter,
            learning_rate,
        )
        name = full_name

        init_cov = init_cov * jnp.ones(len(init_mean))
        if isinstance(process_noise, (int, float)):
            process_noise = jax.tree_map(lambda x: process_noise, init_cov)
        if linplugin:
            _update_fn = staticmethod(update_ldg_reparam_bbb)
        else:
            _update_fn = staticmethod(update_dg_reparam_bbb)

        def init_fn() -> AgentState:
            return staticmethod(init_bbb)(init_mean, init_cov)

        def pred_fn(state: AgentState) -> AgentState:
            return staticmethod(predict_bbb)(state, dynamics_decay, process_noise)

        def update_fn(
            rng_key: PRNGKey, state: AgentState, x: ArrayLike, y: ArrayLike
        ) -> AgentState:
            @jax.jit
            def _step(curr_state, t):
                key = jr.fold_in(rng_key, t)
                new_state = _update_fn(
                    key,
                    state,
                    curr_state,
                    x,
                    y,
                    log_likelihood,
                    emission_mean_function,
                    emission_cov_function,
                    num_samples,
                    empirical_fisher,
                    learning_rate,
                )
                return new_state, new_state

            new_state, _ = jax.lax.scan(_step, state, jnp.arange(num_iter))
            return new_state

        return RebayesAlgorithm(
            init_fn, pred_fn, update_fn, cls.sample, name, full_name
        )
