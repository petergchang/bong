import argparse
from functools import partial
from pathlib import Path
import time
import json
import pandas as pd

import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.optimize import minimize
import matplotlib.pyplot as plt
import optax
import tensorflow_probability.substrates.jax as tfp


from bong.util import run_rebayes_algorithm, gaussian_kl_div, MLP
from bong.src import bbb, blr, bog, bong, experiment_utils

def make_model(args, data):
    assert args.dataset == "reg" # TODO
    model = args.model[0:3] #lin_1, mlp_10_10_1
    if model == "lin":
        init_kwargs, callback, _ = make_model_reg_lin(args, data)
    elif dgp == "mlp":
        init_kwargs, callback, _ = make_model_reg_mlp(args, data)
    else:
        raise Exception(f'Unknown model {args.model}')
    return init_kwargs, callback

def compute_prior_post_linreg(args, data):
    # Compute (batch) true posterior
    noise_std = args.emission_noise
    d = args.data_dim
    mu0, cov0 = jnp.ones(d), jnp.eye(d) # Prior moments
    inv_cov0 = jnp.linalg.inv(cov0)
    cov_post = jnp.linalg.inv(inv_cov0 + data['X_tr'].T @ data['X_tr'] / noise_std**2)
    mu_post = cov_post @ (inv_cov0 @ mu0 + data['X_tr'].T @ data['Y_tr'] / noise_std**2)
    post = {'mu': mu_post, 'cov': cov_post}
    prior = {'mu': mu0, 'cov': 1.0}
    return prior, post


def  make_model_reg_lin(args, data):
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

    # output = transform(key, rebayes_algorithm, pred_state, x, y)
    def callback(key, alg, state, x, y, X_cb=data['X_te'], Y_cb=data['Y_te'], n_samples_mc_nlpd=100):
        curr_cov = state.cov
        if curr_cov.ndim == 1:
            curr_cov = jnp.diag(state.cov)
        kl_div = gaussian_kl_div(post['mu'], post['cov'], state.mean, curr_cov)
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
