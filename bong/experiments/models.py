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

from bong.util import run_rebayes_algorithm, gaussian_kl_div, MLP, make_neuron_str
from bong.src import bbb, blr, bog, bong, experiment_utils


def make_model(args, data):
    if args.dataset == "reg":
        if args.model_type == "lin":
            model = make_lin_reg(args, data)
        elif args.model_type == "mlp":
            model = make_mlp_reg(args, data)
        else:
            raise Exception(f'Unknown model {args.model_type}')
    elif args.dataset == "cls":
        if args.model_type == "lin":
            model = make_lin_cls(args, data)
        elif args.model_type == "mlp":
            model = make_mlp_cls(args, data)
        else:
            raise Exception(f'Unknown model {args.model_type}')
    else:
        raise Exception(f'Unknown dataset {args.dataset}') # ToDO
    return model

#########

def  make_lin_reg(args, data):
    d = args.data_dim
    mu0 = jnp.zeros(d)
    cov0 = args.init_var
    post = compute_post_linreg(args, data, mu0, cov0*jnp.eye(d))
    noise_std = args.emission_noise
    name = 'lin_1'

    log_likelihood = lambda mean, cov, y: \
        jax.scipy.stats.norm.logpdf(y, mean, jnp.sqrt(jnp.diag(cov))).sum()
    em_function = lambda w, x: w @ x
    ec_function = lambda w, x: noise_std * jnp.eye(1)

    model_kwargs = {
        "init_mean": mu0,
        "init_cov": cov0,
        "log_likelihood": log_likelihood,
        "emission_mean_function": em_function,
        "emission_cov_function": ec_function,
        }


    callback = partial(callback_reg, X_te=data['X_te'], Y_te=data['Y_te'],
                X_val=data['X_val'], Y_val=data['Y_val'], post=None,
        em_function = em_function, ec_function = ec_function, log_likelihood = log_likelihood)

    
    def process_callback(output):
        nll_te, nll_val, nlpd_te, nlpd_val, kldiv = output
        s_val = f"Val NLL {nll_val[-1]:.4f},  NLPD: {nlpd_val[-1]:.4f}"
        s_te = f"Test NLL: {nll_te[-1]:.4f},  NLPD: {nlpd_te[-1]:.4f}"
        summary = s_te + "\n" + s_val 
        results = {'nll': nll_te, 'nlpd': nlpd_te, 'nll_val': nll_val, 'nlpd_val': nlpd_val}
        return results, summary

        
    def tune_kl_loss_fn(key, alg, state):
        return gaussian_kl_div(post['mu'], post['cov'], state.mean, state.cov)
    
    d = {'model_kwargs': model_kwargs, 'callback': callback,
        'process_callback': process_callback, 'tune_fn': tune_kl_loss_fn,
        'name': name}
    return d

def compute_post_linreg(args, data, mu0, cov0):
    # Compute (batch) true posterior on training set
    noise_std = args.emission_noise
    inv_cov0 = jnp.linalg.inv(cov0)
    cov_post = jnp.linalg.inv(inv_cov0 + data['X_tr'].T @ data['X_tr'] / noise_std**2)
    mu_post = cov_post @ (inv_cov0 @ mu0 + data['X_tr'].T @ data['Y_tr'] / noise_std**2)
    post = {'mu': mu_post, 'cov': cov_post}
    return post

# output = transform(key, rebayes_algorithm, pred_state, x, y)
def callback_reg(key, alg, state, x, y, X_te, Y_te, X_val, Y_val,
        post, em_function, ec_function, log_likelihood,
        n_samples_mc_nlpd=100):
 
    # Plugin-NLL
    def _nll(curr_mean, xcb, ycb):
        em = em_function(curr_mean, xcb)
        ec = ec_function(curr_mean, xcb)
        return -log_likelihood(em, ec, ycb)
    nll_te = jnp.mean(jax.vmap(_nll, (None, 0, 0))(state.mean, X_te, Y_te))
    nll_val = jnp.mean(jax.vmap(_nll, (None, 0, 0))(state.mean, X_val, Y_val))

    # MC-NLPD
    means = alg.sample(key, state, n_samples_mc_nlpd)
    nlpd_te = jnp.mean(jax.vmap(
        jax.vmap(_nll, (None, 0, 0)), (0, None, None)
    )(means, X_te, Y_te))

    means = alg.sample(key, state, n_samples_mc_nlpd)
    nlpd_val = jnp.mean(jax.vmap(
        jax.vmap(_nll, (None, 0, 0)), (0, None, None)
    )(means, X_val, Y_val))

    # KL
    if post is not None:
        curr_cov = state.cov
        if curr_cov.ndim == 1:
            curr_cov = jnp.diag(state.cov)
        kl_div = gaussian_kl_div(post['mu'], post['cov'], state.mean, curr_cov)
    else:
        kl_div = None

    return nll_te, nll_val, nlpd_te, nlpd_val, kl_div



#######


def make_mlp_reg(args, data):
    neurons = args.model_neurons
    name = f'mlp_{make_neuron_str(neurons)}'
    
    model_kwargs, key = initialize_mlp_model_reg(args.algo_key, neurons,
                        data['X_tr'][0], args.init_var, args.emission_noise)
    em_function = model_kwargs["emission_mean_function"]
    ec_function = model_kwargs["emission_cov_function"]
    log_likelihood = model_kwargs["log_likelihood"]

    callback = partial(callback_reg, X_te=data['X_te'], Y_te=data['Y_te'],
                X_val=data['X_val'], Y_val=data['Y_val'], post=None,
        em_function = em_function, ec_function = ec_function, log_likelihood = log_likelihood)

    def process_callback(output):
        nll_te, nll_val, nlpd_te, nlpd_val, kldiv = output
        s_val = f"Val NLL {nll_val[-1]:.4f},  NLPD: {nlpd_val[-1]:.4f}"
        s_te = f"Test NLL: {nll_te[-1]:.4f},  NLPD: {nlpd_te[-1]:.4f}"
        summary = s_te + "\n" + s_val 
        results = {'nll': nll_te, 'nlpd': nlpd_te, 'nll_val': nll_val, 'nlpd_val': nlpd_val}
        return results, summary

    d = {'model_kwargs': model_kwargs, 'callback': callback,
        'process_callback': process_callback, 'name': name}
    return d

def initialize_mlp_model_reg(key, features, x, init_var, emission_noise):
    if isinstance(key, int):
        key = jr.key(key)
    key, subkey = jr.split(key)
    model = MLP(features=features)
    params = model.init(key, x)
    flat_params, unflatten_fn = ravel_pytree(params)
    apply_fn = lambda w, x: model.apply(unflatten_fn(w), jnp.atleast_1d(x))
    
    noise_std = emission_noise
    log_likelihood = lambda mean, cov, y: \
        jax.scipy.stats.norm.logpdf(y, mean, jnp.sqrt(jnp.diag(cov))).sum()
    em_function = apply_fn # lambda w, x: w @ x
    ec_function = lambda w, x: noise_std * jnp.eye(1)

    d = len(flat_params)
    init_kwargs = {
        "init_mean": flat_params,
        "init_cov": init_var, # optoonally tune
        "log_likelihood": log_likelihood,
        "emission_mean_function": em_function,
        "emission_cov_function": ec_function,
    }
    return init_kwargs, subkey

#############

def  make_lin_cls(args, data):
    raise Exception('TODO')

#######

def make_mlp_cls(args, data):
    raise Exception('TODO')
    neurons = args.model_neurons
    name = f'mlp_{make_neuron_str(neurons)}'
    
    model_kwargs, key = \
            initialize_mlp_model_cls(args.algo_key, neurons, data['X_tr'][0])
    em_function = init_kwargs["emission_mean_function"]
    callback = partial(callback_cls, em_function=em_function,
                         X_cb=data['X_te'], y_cb=data['Y_te'])
    d = {'model_kwargs': model_kwargs, 'callback': callback, 'tune_fn': None, 'name': name}
    return d
    
def initialize_mlp_model_cls(key, features, x):
    if isinstance(key, int):
        key = jr.key(key)
    key, subkey = jr.split(key)
    model = MLP(features=features)
    params = model.init(key, x)
    flat_params, unflatten_fn = ravel_pytree(params)
    apply_fn = lambda w, x: model.apply(unflatten_fn(w), jnp.atleast_1d(x))
    log_likelihood = lambda mean, cov, y: -optax.softmax_cross_entropy(mean, y)
    em_function = apply_fn
    em_linpi_function = lambda w, x: jax.nn.softmax(apply_fn(w, x))
    def ec_function(w, x):
        ps = em_linpi_function(w, x)
        cov = jnp.diag(ps) - jnp.outer(ps, ps) + 1e-5 * jnp.eye(len(ps))
        return jnp.atleast_2d(cov)
    init_kwargs = {
        "init_mean": flat_params,
        "log_likelihood": log_likelihood,
        "emission_mean_function": em_function,
        "em_linpi_function": em_linpi_function,
        "emission_cov_function": ec_function,
    }
    return init_kwargs, subkey

def loss_fn_cls(key, alg, state, em_function, X_val, y_val):
    y_pred_logits = jax.vmap(em_function, (None, 0))(state.mean, X_val)
    negloglikhood = optax.softmax_cross_entropy_with_integer_labels(
        y_pred_logits, y_val
    )
    return jnp.mean(negloglikhood)

def callback_cls(key, alg, state, x, y, em_function, X_cb, y_cb, num_samples=10):
    # Plugin-LL
    ypi_pred_logits = jax.vmap(em_function, (None, 0))(state.mean, X_cb)
    ll_pi = jnp.mean(-optax.softmax_cross_entropy_with_integer_labels(
        ypi_pred_logits, y_cb
    ))
    
    # Plugin-accuracy
    ypi_preds = jnp.argmax(ypi_pred_logits, axis=-1)
    acc_pi = jnp.mean(ypi_preds == y_cb)
    
    # NLPD-LL
    states = alg.sample(key, state, num_samples)
    y_pred_logits = jnp.mean(jax.vmap(
        jax.vmap(em_function, (None, 0)), (0, None)
    )(states, X_cb), axis=0)
    ll = jnp.mean(-optax.softmax_cross_entropy_with_integer_labels(
        y_pred_logits, y_cb
    ))
    
    # NLPD-accuracy
    y_preds = jnp.argmax(y_pred_logits, axis=-1)
    acc = jnp.mean(y_preds == y_cb)
    return ll_pi, acc_pi, ll, acc

