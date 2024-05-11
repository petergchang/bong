
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
from jax.flatten_util import ravel_pytree

from bong.util import run_rebayes_algorithm, gaussian_kl_div, MLP
from bong.src import bbb, blr, bog, bong, experiment_utils



def generate_xdata_mvn(keyroot, N, d, c=1., scale=1):
    key0, key1, keyroot = jr.split(keyroot, 3)
    mean = jnp.zeros(d)
    cov = experiment_utils.generate_covariance_matrix(key0, d, c, scale)
    X = jr.multivariate_normal(key1, mean, cov, (N,))
    return X



def generate_mlp(keyroot, d, nneurons = [1,]):
    # nneurons=[1,] refers to scalar output with no hidden units
    key, keyroot = jr.split(keyroot)
    model = MLP(features = nneurons, use_bias=True)
    params = model.init(key, jnp.ones(d,))
    flat_params, unflatten_fn = ravel_pytree(params)
    apply_fn = lambda w, x: model.apply(unflatten_fn(w), jnp.atleast_1d(x))
    pred_fn = lambda x: model.apply(params, jnp.atleast_1d(x))
    model_dict = {'model': model, 'params': params, 'flat_params': flat_params, 'apply_fn': apply_fn, 'pred_fn': pred_fn}
    return model_dict


def generate_ydata_mlpreg(keyroot, X, model, noise_std=1.0):
    key, keyroot = jr.split(keyroot)
    N, d = X.shape
    predictor = model['pred_fn']
    Ymean = jax.vmap(predictor)(X)
    Y = Ymean + jr.normal(key, (N,1)) * noise_std
    return Y




def make_data_mlp(args):
    d, noise_std = args.data_dim, args.emission_noise
    keyroot = jr.PRNGKey(args.data_key)
    key1, keyroot = jr.split(keyroot)
    model = generate_mlp(key1, args.data_dim, args.data_neurons)

    key1, key2, key3, keyroot = jr.split(keyroot, 4)
    X_tr = generate_xdata_mvn(key1, args.ntrain, d)
    Y_tr = generate_ydata_mlpreg(key1, X_tr, model, noise_std)
    X_val = generate_xdata_mvn(key2, args.nval, d)
    Y_val = generate_ydata_mlpreg(key2, X_val, model, noise_std)
    X_te = generate_xdata_mvn(key3, args.ntest, d)
    Y_te = generate_ydata_mlpreg(key3, X_te, model, noise_std)

    neuron_str = '-'.join(str(num) for num in args.data_neurons)
    name = f'mlpreg-dim{args.data_dim}-neurons{neuron_str}-key{args.data_key}'
    data = {'X_tr': X_tr, 'Y_tr': Y_tr, 'X_val': X_val, 'Y_val': Y_val, 'X_te': X_te, 'Y_te': Y_te, 'name': name}
    return data

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


def init_mlpreg(args, data):
    print("WIP******************")
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

def make_mlpreg(args):
    data = make_data_mlpreg(args)
    init_kwargs, callback, tune_obj_fn = init_mlpreg(args, data)
    return data, init_kwargs, callback, tune_obj_fn