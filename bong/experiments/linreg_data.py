
import argparse
from functools import partial
from pathlib import Path
import time
import json
import pandas as pd
import jax.numpy as jnp
import numpy as np
import jax.random as jr
import jax

from bong.util import run_rebayes_algorithm, gaussian_kl_div
from bong.src import bbb, blr, bog, bong, experiment_utils



def generate_linreg_dataset(
    key, N, d, c=1., scale=1., noise_std=1.0, theta=None
):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    keys = jr.split(key, 4)
    mean = jnp.zeros(d)
    cov = experiment_utils.generate_covariance_matrix(keys[0], d, c, scale)
    X = jr.multivariate_normal(keys[1], mean, cov, (N,))
    if theta is None:
        theta = jr.uniform(keys[2], (d,), minval=-1., maxval=1.)
        theta = theta / jnp.linalg.norm(theta)
    Y = X @ theta + jr.normal(keys[3], (N,)) * noise_std
    return X, Y, theta



def make_data_linreg(args):
     # Generate dataset
    key = jr.PRNGKey(args.data_key)
    key1, key2, key3, subkey = jr.split(key, 4)
    d, noise_std = args.data_dim, args.emission_noise
    X_tr, Y_tr, theta = generate_linreg_dataset(
        key1, args.ntrain, d, noise_std=noise_std
    )
    X_val, Y_val, _ = generate_linreg_dataset(
        key2, args.nval, d, noise_std=noise_std, theta=theta
    )
    X_te, Y_te, _ = generate_linreg_dataset(
        key3, args.ntest, d, noise_std=noise_std, theta=theta
    )
    name = f'linreg-dim{args.data_dim}-key{args.data_key}'
    data = {'X_tr': X_tr, 'Y_tr': Y_tr, 'X_val': X_val, 'Y_val': Y_val, 'X_te': X_te, 'Y_te': Y_te, 'name': name}
    return data, subkey

def compute_prior_post_linreg(args, data):
    # Compute (batch) true posterior
    noise_std = args.emission_noise
    d = args.data_dim
    mu0, cov0 = jnp.ones(d), jnp.eye(d) # Prior moments
    inv_cov0 = jnp.linalg.inv(cov0)
    cov_post = jnp.linalg.inv(inv_cov0 + data['X_tr'].T @ data['X_tr'] / noise_std**2)
    mu_post = cov_post @ (inv_cov0 @ mu0 + data['X_tr'].T @ data['Y_tr'] / noise_std**2)
    post = {'mu': mu_post, 'cov': cov_post}
    prior = {'mu': mu0, 'cov': cov0}
    return prior, post


def init_linreg(args, data):
    prior, post = compute_prior_post_linreg(args, data) # training set
    noise_std = args.emission_noise
    
    # Compute KL divergence, plugin NLL, MC-NLPD with agents
    log_likelihood = lambda mean, cov, y: \
        jax.scipy.stats.norm.logpdf(y, mean, jnp.sqrt(jnp.diag(cov))).sum()
    em_function = lambda w, x: w @ x
    ec_function = lambda w, x: noise_std * jnp.eye(1)

    init_kwargs = {
        "init_mean": prior['mu'],
        "init_cov": prior['cov'],
        "log_likelihood": log_likelihood,
        "emission_mean_function": em_function,
        "emission_cov_function": ec_function,
        }

    def callback(key, alg, state, x, y, X_cb=data['X_te'], Y_cb=data['Y_te'], n_samples_mc_nlpd=100):
        # KL-div
        kl_div = gaussian_kl_div(post['mu'], post['cov'], state.mean, state.cov)
        # Plugin-NLL
        def _nll(curr_mean, xcb, ycb):
            em = em_function(curr_mean, xcb)
            ec = ec_function(curr_mean, xcb)
            return -log_likelihood(em, ec, ycb)
        nll = jnp.mean(jax.vmap(_nll, (None, 0, 0))(state.mean, X_cb, Y_cb))
        # MC-NLPD
        means = alg.sample(key, state, n_samples_mc_nlpd)
        nlpd = jnp.mean(jax.vmap(
            jax.vmap(_nll, (None, 0, 0)), (0, None, None)
        )(means, X_cb, Y_cb))
        return kl_div, nll, nlpd
    
    
    def tune_kl_loss_fn(key, alg, state):
        return gaussian_kl_div(post['mu'], post['cov'], state.mean, state.cov)

    return init_kwargs, callback, tune_kl_loss_fn

def make_linreg(args):
    data, subkey = make_data_linreg(args)
    init_kwargs, callback, tune_obj_fn = init_linreg(args, data)
    return data, init_kwargs, callback, tune_obj_fn