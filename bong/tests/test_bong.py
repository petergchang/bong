import math
from typing import Sequence

from flax import linen as nn
import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import jax.random as jr

from bong.src import bong
from bong.util import run_rebayes_algorithm
from dynamax.nonlinear_gaussian_ssm.inference_ekf import (
    extended_kalman_filter
)
from dynamax.nonlinear_gaussian_ssm.models import ParamsNLGSSM


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


def toydata(N=60, std1=1, std2=1.1, key=0):
    """Creates a seperable toy linear classification dataset"""
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    keys = jr.split(key)
    mid_index = math.floor(N/2)
    mu1 = jnp.stack([jnp.ones(mid_index), jnp.ones(N-mid_index)*5], axis=1)
    mu2 = jnp.stack([jnp.ones(mid_index)*+5, jnp.ones(N-mid_index)], axis=1)
    X = jnp.concatenate([
        jr.normal(keys[0], mu1.shape)*std1 + mu1,
        jr.normal(keys[1], mu2.shape)*std2 + mu2])
    y = jnp.ones(N)
    y = y.at[mid_index:N].set(0)
    
    return jnp.array(X), jnp.array(y)


def test_linearized_bong(key=42, num_timesteps=15):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
        
    # Load model and set up logistic regression
    X, y = toydata()
    linear_model = MLP(features = [1,])
    params = linear_model.init(key, jnp.ones(2,))
    flat_params, unflatten_fn = ravel_pytree(params)
    apply_fn = lambda w, x: \
        jax.nn.sigmoid(linear_model.apply(unflatten_fn(w), jnp.atleast_1d(x)))
    init_cov = 0.1 * jnp.eye(len(flat_params))
    dynamics_cov = 0. * jnp.eye(len(flat_params))
    emission_cov = 0.01 * jnp.eye(1)
    
    # Run EKF from dynamax
    ekf_params = ParamsNLGSSM(
        initial_mean = flat_params,
        initial_covariance = init_cov,
        dynamics_function = lambda z, u: z,
        dynamics_covariance = dynamics_cov,
        emission_function = apply_fn,
        emission_covariance = emission_cov
    )
    ekf_post = extended_kalman_filter(ekf_params, y, inputs=X)
    
    # Run EKF using linearized-BONG
    fg_lbong = bong.fg_bong(
        init_mean = flat_params,
        init_cov = init_cov,
        log_likelihood = lambda m, c, y: 0.0,
        emission_mean_function = apply_fn,
        emission_cov_function = lambda w, x: emission_cov,
        linplugin = True,
    )
    bong_post, _ = run_rebayes_algorithm(
        key, fg_lbong, X, y,
    )
    
    assert jnp.allclose(ekf_post.filtered_means[-1], bong_post.mean, atol=1e-2)
    assert jnp.allclose(ekf_post.filtered_covariances[-1], bong_post.cov, 
                        atol=1e-2)