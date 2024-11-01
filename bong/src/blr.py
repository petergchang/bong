from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr

from bong.base import RebayesAlgorithm
from bong.src.bong import sample_dg_bong, sample_dlrg_bong, sample_fg_bong
from bong.src.states import AgentState, DLRAgentState
from bong.custom_types import ArrayLike, PRNGKey
from bong.util import fast_svd, hess_diag_approx, make_full_name


def init_blr(
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


def init_blr_dlr(
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


def predict_blr(
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


def predict_blr_dlr(
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


def update_fg_blr(
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
    new_prec = (1 - learning_rate) * prec + learning_rate * prec0 - learning_rate * hess
    new_cov = jnp.linalg.pinv(new_prec)
    new_mean = (
        mean
        + learning_rate * new_cov @ prec0 @ (mean0 - mean)
        + learning_rate * new_cov @ g
    )
    new_state = AgentState(new_mean, new_cov)
    return new_state


def update_lfg_blr(
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
        learning_rate: Learning rate for the update.

    Returns:
        Updated belief state.
    """
    mean0, cov0 = state_pred
    mean, cov = state
    R = jnp.atleast_2d(emission_cov_function(mean, x))
    if empirical_fisher:
        R_inv = jnp.linalg.lstsq(R, jnp.eye(R.shape[0]))[0]

        def ll_fn(params):
            y_pred = emission_mean_function(params, x)
            return -0.5 * (y - y_pred).T @ R_inv @ (y - y_pred)

        grad = jax.grad(ll_fn)(mean)
        G = -jnp.outer(grad, grad)
        prec0, prec = jnp.linalg.pinv(cov0), jnp.linalg.pinv(cov)
        new_prec = (
            (1 - learning_rate) * prec + learning_rate * prec0 - learning_rate * G
        )
        new_cov = jnp.linalg.pinv(new_prec)
        new_mean = (
            mean
            + learning_rate * new_cov @ prec0 @ (mean0 - mean)
            + learning_rate * new_cov @ grad
        )
    else:
        y_pred = jnp.atleast_1d(emission_mean_function(mean, x))
        H = jnp.atleast_2d(jax.jacrev(emission_mean_function)(mean, x))
        R_inv = jnp.linalg.pinv(R)
        prec0, prec = jnp.linalg.pinv(cov0), jnp.linalg.pinv(cov)
        new_prec = (
            (1 - learning_rate) * prec
            + learning_rate * prec0
            + learning_rate * H.T @ R_inv @ H
        )
        new_cov = jnp.linalg.pinv(new_prec)
        new_mean = (
            mean
            + learning_rate * new_cov @ prec0 @ (mean0 - mean)
            + learning_rate * new_cov @ H.T @ R_inv @ (y - y_pred)
        )
    new_state = AgentState(new_mean, new_cov)
    return new_state


def update_dlrg_blr(
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
    if empirical_fisher:
        prec_lr_update = jnp.sqrt(learning_rate / num_samples) * grads.T
        prec_lr_tilde = jnp.hstack(
            [
                jnp.sqrt(1 - learning_rate) * prec_lr,
                jnp.sqrt(learning_rate) * prec_lr0,
                prec_lr_update.reshape(P, -1),
            ]
        )
    else:
        raise NotImplementedError
    _, L_tilde = prec_lr_tilde.shape
    prec_diag_tilde = (1 - learning_rate) * prec_diag + learning_rate * prec_diag0
    G = jnp.linalg.pinv(
        jnp.eye(L_tilde) + prec_lr_tilde.T @ (prec_lr_tilde / prec_diag_tilde)
    )
    mean_term1 = (
        (prec_diag0 / prec_diag_tilde).ravel() * (mean0 - mean)
        + (prec_lr0 / prec_diag_tilde) @ (prec_lr0.T @ (mean0 - mean))
        + 1 / (prec_diag_tilde.ravel()) * g
    )
    mean_term2 = (prec_lr_tilde / prec_diag_tilde @ G) @ (
        (prec_lr_tilde * (prec_diag0 / prec_diag_tilde)).T @ (mean0 - mean)
        + (prec_lr_tilde / prec_diag_tilde).T @ prec_lr0 @ (prec_lr0.T @ (mean0 - mean))
        + (prec_lr_tilde / prec_diag_tilde).T @ g
    )
    mean_update = mean_term1 - mean_term2
    new_mean = mean + learning_rate * mean_update
    U, Lamb = fast_svd(prec_lr_tilde)
    U_new, Lamb_new = U[:, :L], Lamb[:L]
    U_extra, Lamb_extra = U[:, L:], Lamb[L:]
    extra_prec_lr = Lamb_extra * U_extra
    new_prec_lr = Lamb_new * U_new
    new_prec_diag = (
        prec_diag_tilde
        + jnp.einsum("ij,ij->i", extra_prec_lr, extra_prec_lr)[:, jnp.newaxis]
    )
    new_state = DLRAgentState(new_mean, new_prec_diag, new_prec_lr)
    return new_state


def update_ldlrg_blr(
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
    R = jnp.atleast_2d(emission_cov_function(mean, x))
    if empirical_fisher:
        R_inv = jnp.linalg.lstsq(R, jnp.eye(R.shape[0]))[0]

        def ll_fn(params):
            y_pred = emission_mean_function(params, x)
            return -0.5 * (y - y_pred).T @ R_inv @ (y - y_pred)

        grad = jax.grad(ll_fn)(mean)
        prec_lr_update = jnp.sqrt(learning_rate) * grad
        prec_lr_tilde = jnp.hstack(
            [
                jnp.sqrt(1 - learning_rate) * prec_lr,
                jnp.sqrt(learning_rate) * prec_lr0,
                prec_lr_update.reshape(P, -1),
            ]
        )
        _, L_tilde = prec_lr_tilde.shape
        prec_diag_tilde = (1 - learning_rate) * prec_diag + learning_rate * prec_diag0
        G = jnp.linalg.pinv(
            jnp.eye(L_tilde) + prec_lr_tilde.T @ (prec_lr_tilde / prec_diag_tilde)
        )
        mean_term1 = (
            (prec_diag0 / prec_diag_tilde).ravel() * (mean0 - mean)
            + (prec_lr0 / prec_diag_tilde) @ (prec_lr0.T @ (mean0 - mean))
            + 1 / (prec_diag_tilde.ravel()) * grad
        )
        mean_term2 = (prec_lr_tilde / prec_diag_tilde @ G) @ (
            (prec_lr_tilde * (prec_diag0 / prec_diag_tilde)).T @ (mean0 - mean)
            + (prec_lr_tilde / prec_diag_tilde).T
            @ prec_lr0
            @ (prec_lr0.T @ (mean0 - mean))
            + (prec_lr_tilde / prec_diag_tilde).T @ grad
        )
        mean_update = mean_term1 - mean_term2
        new_mean = mean + learning_rate * mean_update
        U, Lamb = fast_svd(prec_lr_tilde)
        U_new, Lamb_new = U[:, :L], Lamb[:L]
        U_extra, Lamb_extra = U[:, L:], Lamb[L:]
        extra_prec_lr = Lamb_extra * U_extra
        new_prec_lr = Lamb_new * U_new
        new_prec_diag = (
            prec_diag_tilde
            + jnp.einsum("ij,ij->i", extra_prec_lr, extra_prec_lr)[:, jnp.newaxis]
        )
    else:
        y_pred = jnp.atleast_1d(emission_mean_function(mean, x))
        H = jnp.atleast_2d(jax.jacrev(emission_mean_function)(mean, x))
        R_chol = jnp.linalg.cholesky(R)
        A = jnp.linalg.lstsq(R_chol, jnp.eye(R.shape[0]))[0].T
        prec_lr_tilde = jnp.hstack(
            [
                jnp.sqrt(1 - learning_rate) * prec_lr,
                jnp.sqrt(learning_rate) * prec_lr0,
                jnp.sqrt(learning_rate) * (H.T @ A).reshape(P, -1),
            ]
        )
        _, L_tilde = prec_lr_tilde.shape
        prec_diag_tilde = (1 - learning_rate) * prec_diag + learning_rate * prec_diag0
        G = jnp.linalg.pinv(
            jnp.eye(L_tilde) + prec_lr_tilde.T @ (prec_lr_tilde / prec_diag_tilde)
        )
        mean_term1 = (
            (prec_diag0 / prec_diag_tilde).ravel() * (mean0 - mean)
            + (prec_lr0 / prec_diag_tilde) @ (prec_lr0.T @ (mean0 - mean))
            + (H.T @ A) @ A.T / prec_diag_tilde @ (y - y_pred)
        )
        mean_term2 = (prec_lr_tilde / prec_diag_tilde @ G) @ (
            (prec_lr_tilde * (prec_diag0 / prec_diag_tilde)).T @ (mean0 - mean)
            + (prec_lr_tilde / prec_diag_tilde).T
            @ prec_lr0
            @ (prec_lr0.T @ (mean0 - mean))
            + (prec_lr_tilde / prec_diag_tilde).T @ (H.T @ A) @ A.T @ (y - y_pred)
        )
        mean_update = mean_term1 - mean_term2
        new_mean = mean + learning_rate * mean_update
        U, Lamb = fast_svd(prec_lr_tilde)
        U_new, Lamb_new = U[:, :L], Lamb[:L]
        U_extra, Lamb_extra = U[:, L:], Lamb[L:]
        extra_prec_lr = Lamb_extra * U_extra
        new_prec_lr = Lamb_new * U_new
        new_prec_diag = (
            prec_diag_tilde
            + jnp.einsum("ij,ij->i", extra_prec_lr, extra_prec_lr)[:, jnp.newaxis]
        )
    new_state = DLRAgentState(new_mean, new_prec_diag, new_prec_lr)
    return new_state


def update_dg_blr(
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
    grad_est = jnp.mean(grads, axis=0)
    if empirical_fisher:
        hess_diag = -1 / num_samples * jnp.einsum("ij,ij->j", grads, grads)
    else:

        def hess_diag_fn(param):
            return hess_diag_approx(keys[1], ll_fn, param)

        hess_diag = jnp.mean(jax.vmap(hess_diag_fn)(z), axis=0)
    prec0, prec = 1 / cov0, 1 / cov
    new_prec = (
        (1 - learning_rate) * prec + learning_rate * prec0 - learning_rate * hess_diag
    )
    new_cov = 1 / new_prec
    new_mean = (
        mean
        + learning_rate * new_cov * prec0 * (mean0 - mean)
        + learning_rate * new_cov * grad_est
    )
    new_state = AgentState(new_mean, new_cov)
    return new_state


def update_ldg_blr(
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
    """Update the linearized-plugin diagonal-covariance Gaussian
    belief state with a new observation.

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
    mean0, cov0 = state_pred
    mean, cov = state
    R = jnp.atleast_2d(emission_cov_function(mean, x))
    R_inv = jnp.linalg.lstsq(R, jnp.eye(R.shape[0]))[0]
    if empirical_fisher:

        def ll_fn(params):
            y_pred = emission_mean_function(params, x)
            return -0.5 * (y - y_pred).T @ R_inv @ (y - y_pred)

        grad = jax.grad(ll_fn)(mean)
        G_diag = -(grad**2)
        prec0, prec = 1 / cov0, 1 / cov
        new_prec = (
            (1 - learning_rate) * prec + learning_rate * prec0 - learning_rate * G_diag
        )
        new_cov = 1 / new_prec
        new_mean = (
            mean
            + learning_rate * new_cov * prec0 * (mean0 - mean)
            + learning_rate * new_cov * grad
        )
    else:
        y_pred = jnp.atleast_1d(emission_mean_function(mean, x))
        H = jnp.atleast_2d(jax.jacrev(emission_mean_function)(mean, x))
        prec0, prec = 1 / cov0, 1 / cov
        new_prec = (
            (1 - learning_rate) * prec
            + learning_rate * prec0
            + learning_rate * ((H.T @ R_inv) * H.T).sum(-1)
        )
        new_cov = 1 / new_prec
        new_mean = (
            mean
            + learning_rate * new_cov * prec0 * (mean0 - mean)
            + learning_rate * new_cov * (H.T @ R_inv @ (y - y_pred))
        )
    new_state = AgentState(new_mean, new_cov)
    return new_state


def update_fg_reparam_blr(
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
    under the reparameterized BLR model.

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
    new_cov = cov + learning_rate * cov @ (hess + prec - prec0) @ cov
    mean_update = g + prec0 @ (mean0 - mean)
    new_mean = mean + learning_rate * cov @ mean_update
    new_state = AgentState(new_mean, new_cov)
    return new_state


def update_lfg_reparam_blr(
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
    with a new observation under the reparameterized BLR model.

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
    mean0, cov0 = state_pred
    mean, cov = state
    R = jnp.atleast_2d(emission_cov_function(mean, x))
    R_inv = jnp.linalg.pinv(R)
    if empirical_fisher:

        def ll_fn(params):
            y_pred = emission_mean_function(params, x)
            return -0.5 * (y - y_pred).T @ R_inv @ (y - y_pred)

        grad = jax.grad(ll_fn)(mean)
        G = -jnp.outer(grad, grad)
        prec0, prec = jnp.linalg.pinv(cov0), jnp.linalg.pinv(cov)
        new_cov = cov + learning_rate * cov @ (G + prec - prec0) @ cov
        mean_update = grad + prec0 @ (mean0 - mean)
        new_mean = mean + learning_rate * cov @ mean_update
    else:
        y_pred = jnp.atleast_1d(emission_mean_function(mean, x))
        H = jnp.atleast_2d(jax.jacrev(emission_mean_function)(mean, x))
        prec0, prec = jnp.linalg.pinv(cov0), jnp.linalg.pinv(cov)
        new_cov = cov + learning_rate * cov @ (H.T @ R_inv @ H + prec - prec0) @ cov
        mean_update = H.T @ R_inv @ (y - y_pred) + prec0 @ (mean0 - mean)
        new_mean = mean + learning_rate * cov @ mean_update
    new_state = AgentState(new_mean, new_cov)
    return new_state


def update_dg_reparam_blr(
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
    with a new observation under the reparameterized BLR model.

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
    if empirical_fisher:
        hess_diag = -1 / num_samples * jnp.einsum("ij,ij->j", grads, grads)
    else:

        def hess_diag_fn(param):
            return hess_diag_approx(keys[1], ll_fn, param)

        hess_diag = jnp.mean(jax.vmap(hess_diag_fn)(z), axis=0)
    g = jnp.mean(grads, axis=0)
    prec0, prec = 1 / cov0, 1 / cov
    new_cov = cov + learning_rate * cov * (hess_diag + prec - prec0) * cov
    mean_update = g + prec0 * (mean0 - mean)
    new_mean = mean + learning_rate * cov * mean_update
    new_state = AgentState(new_mean, new_cov)
    return new_state


def update_ldg_reparam_blr(
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
    """Update the linearized-plugin diagonal-covariance Gaussian
    belief state with a new observation under the reparameterized BONG model.

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
    mean0, cov0 = state_pred
    mean, cov = state
    R = jnp.atleast_2d(emission_cov_function(mean, x))
    if empirical_fisher:
        R_inv = jnp.linalg.lstsq(R, jnp.eye(R.shape[0]))[0]

        def ll_fn(params):
            y_pred = emission_mean_function(params, x)
            return -0.5 * (y - y_pred).T @ R_inv @ (y - y_pred)

        grad = jax.grad(ll_fn)(mean)
        G_diag = -(grad**2)
        prec0, prec = 1 / cov0, 1 / cov
        new_cov = cov + learning_rate * cov * (G_diag + prec - prec0) * cov
        mean_update = grad + prec0 * (mean0 - mean)
        new_mean = mean + learning_rate * cov * mean_update
    else:
        y_pred = jnp.atleast_1d(emission_mean_function(mean, x))
        H = jnp.atleast_2d(jax.jacrev(emission_mean_function)(mean, x))
        R = jnp.atleast_2d(emission_cov_function(mean, x))
        HTRinv = jnp.linalg.lstsq(R, H)[0].T
        prec0, prec = 1 / cov0, 1 / cov
        new_cov = (
            cov + learning_rate * cov * ((HTRinv * H.T).sum(-1) - prec0 + prec) * cov
        )
        mean_update = HTRinv @ (y - y_pred) + prec0 * (mean0 - mean)
        new_mean = mean + learning_rate * cov * mean_update
    new_state = AgentState(new_mean, new_cov)
    return new_state


class fg_blr:
    """Full-covariance Gaussian BLR algorithm.

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
            "blr",
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
            _update_fn = staticmethod(update_lfg_blr)
        else:
            _update_fn = staticmethod(update_fg_blr)

        def init_fn() -> AgentState:
            return staticmethod(init_blr)(init_mean, init_cov)

        def pred_fn(state: AgentState) -> AgentState:
            return staticmethod(predict_blr)(state, dynamics_decay, process_noise)

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


class dlrg_blr:
    """DLR-precision Gaussian BLR algorithm.

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
            "blr",
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
            _update_fn = staticmethod(update_ldlrg_blr)
        else:
            _update_fn = staticmethod(update_dlrg_blr)

        def init_fn() -> DLRAgentState:
            return staticmethod(init_blr_dlr)(init_mean, init_prec_diag, init_lr)

        def pred_fn(state: DLRAgentState) -> DLRAgentState:
            return staticmethod(predict_blr_dlr)(state, dynamics_decay, process_noise)

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


class dg_blr:
    """Diagonal-covariance Gaussian BLR algorithm.

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
            "blr",
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
            _update_fn = staticmethod(update_ldg_blr)
        else:
            _update_fn = staticmethod(update_dg_blr)

        def init_fn() -> AgentState:
            return staticmethod(init_blr)(init_mean, init_cov)

        def pred_fn(state: AgentState) -> AgentState:
            return staticmethod(predict_blr)(state, dynamics_decay, process_noise)

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


class fg_reparam_blr:
    """Full-covariance Gaussian reparameterized BLR algorithm.

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
            "blr",
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
            _update_fn = staticmethod(update_lfg_reparam_blr)
        else:
            _update_fn = staticmethod(update_fg_reparam_blr)

        def init_fn() -> AgentState:
            return staticmethod(init_blr)(init_mean, init_cov)

        def pred_fn(state: AgentState) -> AgentState:
            return staticmethod(predict_blr)(state, dynamics_decay, process_noise)

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


class dg_reparam_blr:
    """Diagonal-covariance Gaussian reparameterized BLR algorithm.

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
            "blr",
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
            _update_fn = staticmethod(update_ldg_reparam_blr)
        else:
            _update_fn = staticmethod(update_dg_reparam_blr)

        def init_fn() -> AgentState:
            return staticmethod(init_blr)(init_mean, init_cov)

        def pred_fn(state: AgentState) -> AgentState:
            return staticmethod(predict_blr)(state, dynamics_decay, process_noise)

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
