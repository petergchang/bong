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
from job_utils import  make_neuron_str, parse_neuron_str
from bong.src import bbb, blr, bog, bong, experiment_utils


def make_model(key, args, data):
    if (args.dataset == "reg" or args.dataset == "sarcos"):
        if args.model_type == "lin":
            key, model = make_lin_reg(key, args, data)
        elif args.model_type == "mlp":
            key, model = make_mlp_reg(key, args, data)
        else:
            raise Exception(f'Unknown model {args.model_type}')
    elif args.dataset == "cls":
        if args.model_type == "lin":
            key, model = make_lin_cls(key, args, data) 
        elif args.model_type == "mlp":
            key, model = make_mlp_cls(key, args, data)
        else:
            raise Exception(f'Unknown model {args.model_type}')
    else:
        raise Exception(f'Unknown dataset {args.dataset}') # ToDO
    return key, model

######### Regressiom helpers

gauss_log_likelihood = lambda mean, cov, y: \
    jax.scipy.stats.norm.logpdf(y, mean, jnp.sqrt(jnp.diag(cov))).sum()


def nll_gauss(mu_y, v_y, x, y):
    m = mu_y * jnp.eye(1)
    c = v_y * jnp.eye(1)
    return -gauss_log_likelihood(m, c, y)

def nll_linreg(w, sigma2, x, y):
    m = jnp.dot(w, x) * jnp.eye(1)
    c = sigma2 * jnp.eye(1)
    return -gauss_log_likelihood(m, c, y)

def fit_gauss_baseline(Xtrain, ytrain):
    mu_y, v_y = jnp.mean(ytrain), jnp.var(ytrain)
    return mu_y, v_y


def fit_linreg_baseline(Xtrain, ytrain, method='lstsq'):
    N, D = Xtrain.shape
    if method=='lstsq':
        if N>1000:
            print(f'fit_linreg_baseline is a batch solver, slow with N={N}')
        w, sum_sq_residuals, rank, s = jnp.linalg.lstsq(Xtrain, ytrain, rcond=None) 
        sum_sq_residuals = sum_sq_residuals[0] # convert to scala
    elif method == 'sgd':
        from sklearn.linear_model import SGDRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        sgd = SGDRegressor(max_iter=1000, tol=1e-3, fit_intercept=False) # We assume column of 1s has been added
        pipeline = make_pipeline(StandardScaler(), sgd)
        #pipeline = make_pipeline(sgd) # we assume pre-standardized
        pipeline.fit(Xtrain, ytrain)
        w = pipeline.named_steps['sgdregressor'].coef_
        #bias = pipeline.named_steps['sgdregressor'].intercept_
        assert(len(w)==D)
        yhat = Xtrain @ w
        sum_sq_residuals = jnp.sum(jnp.square(yhat - ytrain))
    else:
        raise Exception(f'Unrecognized method {method}')
    
    sigma2 = sum_sq_residuals/N
    return w, sigma2

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
    key, subkey = jr.split(key)
    means = alg.sample(subkey, state, n_samples_mc_nlpd)
    nlpd_te = jnp.mean(jax.vmap(
        jax.vmap(_nll, (None, 0, 0)), (0, None, None)
    )(means, X_te, Y_te))

    key, subkey = jr.split(key)
    means = alg.sample(subkey, state, n_samples_mc_nlpd)
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

def compute_post_linreg(args, data, mu0, cov0):
    # Compute (batch) true posterior on training set
    noise_std = args.emission_noise
    inv_cov0 = jnp.linalg.inv(cov0)
    cov_post = jnp.linalg.inv(inv_cov0 + data['X_tr'].T @ data['X_tr'] / noise_std**2)
    mu_post = cov_post @ (inv_cov0 @ mu0 + data['X_tr'].T @ data['Y_tr'] / noise_std**2)
    post = {'mu': mu_post, 'cov': cov_post}
    return post

def compute_reg_baselines(args, data):
    N = data['X_tr'].shape[0]
    noise_var = args.emission_noise
    results = {}
    mu_y, sigma2 = fit_gauss_baseline(data['X_tr'], data['Y_tr'])
    results['nll_te_gauss_learned_var'] = jnp.mean(jax.vmap(nll_gauss, (None, None, 0, 0))(mu_y, sigma2, data['X_te'], data['Y_te']))
    results['nll_te_gauss_fixed_var'] = jnp.mean(jax.vmap(nll_gauss, (None, None, 0, 0))(mu_y, noise_var, data['X_te'], data['Y_te']))

    if args.linreg_baseline:
        w, sigma2 = fit_linreg_baseline(data['X_tr'], data['Y_tr'])
        results['nll_te_linreg_learned_var'] = jnp.mean(jax.vmap(nll_linreg, (None, None, 0, 0))(w, sigma2, data['X_te'], data['Y_te']))
        results['nll_te_linreg_fixed_var'] = jnp.mean(jax.vmap(nll_linreg, (None, None, 0, 0))(w, noise_var, data['X_te'], data['Y_te']))
    else:
        results['nll_te_linreg_learned_var'] = 0.0
        results['nll_te_linreg_fixed_var'] = 0.0    

    return  results
       


def  make_lin_reg(key, args, data):
    #d = args.data_dim
    d = data['X_tr'].shape[1] # in case we added a column of 1s
    obs_var = args.emission_noise
    if obs_var == -1:
        obs_var = 0.1*jnp.var(data['Y_tr']) # set approximate scale
    name = f'lin_1[P={d}]'

    mu0 = jnp.zeros(d)
    cov0 = args.init_var
    post = compute_post_linreg(args, data, mu0, cov0*jnp.eye(d))

    em_function = lambda w, x: w @ x
    ec_function = lambda w, x: obs_var * jnp.eye(1) # fixed observation 

    model_kwargs = {
        "init_mean": mu0,
        "init_cov": cov0,
        "log_likelihood": gauss_log_likelihood,
        "emission_mean_function": em_function,
        "emission_cov_function": ec_function,
        }


    callback = partial(callback_reg, X_te=data['X_te'], Y_te=data['Y_te'],
                X_val=data['X_val'], Y_val=data['Y_val'], post=post,
        em_function = em_function, ec_function = ec_function, log_likelihood = gauss_log_likelihood)

    baselines = compute_reg_baselines(args, data)

    
    def process_callback(output):
        nll_te, nll_val, nlpd_te, nlpd_val, kldiv = output
        s_val = f"Val NLL {nll_val[-1]:.4f},  NLPD: {nlpd_val[-1]:.4f}"
        s_te = f"Test NLL: {nll_te[-1]:.4f},  NLPD: {nlpd_te[-1]:.4f}"
        s_kl = f"KL: {kldiv[-1]:0.4f}"
        summary = s_kl + "\n" + s_te + "\n" + s_val 

        results = {'nll': nll_te, 'nlpd': nlpd_te,   
                    'nll_val': nll_val, 'nlpd_val': nlpd_val,
                    'kldiv': kldiv, 'kldiv_val': kldiv,
                    'nlpd_baseline-linreg': baselines['nll_te_linreg_fixed_var'],
        }

        
        return results, summary
        
    def tune_kl_loss_fn(key, alg, state):
        return gaussian_kl_div(post['mu'], post['cov'], state.mean, state.cov)
    
    dct = {
        'model_kwargs': model_kwargs,
        'callback': callback,
        'process_callback': process_callback,
        'tune_fn': tune_kl_loss_fn,
        'name': name,
        'nparams': d,
        }
    return key, dct


#######


def make_mlp_reg(key, args, data):
    neurons = parse_neuron_str(args.model_str)
    obs_var = args.emission_noise
    if obs_var == -1:
        obs_var = 0.1*jnp.var(data['Y_tr']) # set approximate scale
        print('estimated noise variance', obs_var)
    key, subkey = jr.split(key)
    key, model_kwargs = initialize_mlp_model_reg(subkey, neurons,
                        data['X_tr'][0], args.init_var, obs_var, args.use_bias, args.use_bias_layer1)
    nparams = model_kwargs['nparams']
    model_name = f'mlp_{args.model_str}[P={nparams}]'


    em_function = model_kwargs["emission_mean_function"]
    ec_function = model_kwargs["emission_cov_function"]
    log_likelihood = model_kwargs["log_likelihood"]

    callback = partial(callback_reg, X_te=data['X_te'], Y_te=data['Y_te'],
                X_val=data['X_val'], Y_val=data['Y_val'], post=None,
        em_function = em_function, ec_function = ec_function, log_likelihood = gauss_log_likelihood)

    #baselines = compute_reg_baselines(args, data)

    def process_callback(output):
        nll_te, nll_val, nlpd_te, nlpd_val, kldiv = output
        summary = f"NLL-PI: {nll_te[-1]:.4f},  NLPD-MC: {nlpd_te[-1]:.4f}"
        results = {'nll': nll_te, 'nlpd': nlpd_te, 'nll_val': nll_val, 'nlpd_val': nlpd_val,
                    #'nlpd_baseline-linreg': baselines['nll_te_linreg_fixed_var'],
                   }
        return results, summary

    dct = {
        'model_kwargs': model_kwargs,
        'callback': callback,
        'process_callback': process_callback,
        'name': model_name,
        'nparams': model_kwargs['nparams'],
        }
    return key, dct

def initialize_mlp_model_reg(key, features, x, init_var, emission_noise, use_bias=True,  use_bias_layer1=True):
    if isinstance(key, int):
        key = jr.key(key)
    key, subkey = jr.split(key)
    model = MLP(features=features, use_bias=use_bias, use_bias_first_layer=use_bias_layer1)
    params = model.init(subkey, x)
    key, subkey = jr.split(key)
    flat_params, unflatten_fn = ravel_pytree(params)
    apply_fn = lambda w, x: model.apply(unflatten_fn(w), jnp.atleast_1d(x))
    true_pred_fn = lambda x: model.apply(params, jnp.atleast_1d(x))
    
    noise_var = emission_noise
    em_function = apply_fn 
    ec_function = lambda w, x: noise_var * jnp.eye(1)

    d = len(flat_params)
    init_kwargs = {
        "init_mean": flat_params,
        "init_cov": init_var,
        "log_likelihood": gauss_log_likelihood,
        "emission_mean_function": em_function,
        "emission_cov_function": ec_function,
        "nparams": d,
        "true_pred_fn": true_pred_fn
    }
    return key, init_kwargs

#############

def  make_lin_cls(args, data):
    raise Exception('TODO')

#######

    
def make_mlp_cls(key, args, data):
    neurons = parse_neuron_str(args.model_str)
    key, subkey = jr.split(key)
    key, model_kwargs = initialize_mlp_model_cls(subkey, neurons,
                        data['X_tr'][0], args.init_var, args.use_bias)
    nparams = model_kwargs['nparams']
    model_name = f'mlp_{args.model_str}[P={nparams}]'

    em_function = model_kwargs["emission_mean_function"]
    callback = partial(callback_cls, X_te=data['X_te'], Y_te=data['Y_te'],
                X_val=data['X_val'], Y_val=data['Y_val'], em_function = em_function)

    def process_callback(output):
        nll_te, nll_val, acc_pi_te, acc_pi_val, nlpd_te, nlpd_val, acc_te, acc_val = output
        summary1 = f"NLL-PI: {nll_te[-1]:.4f},  NLPD-MC: {nlpd_te[-1]:.4f}"
        summary2 = f"Acc-PI: {acc_pi_te[-1]:.4f}, Acc-MC: {acc_te[-1]:.4f}"
        summary = summary1 + "\n" + summary2
        results = {'nll-pi': nll_te, 'nlpd': nlpd_te, 'acc-pi': acc_pi_te, 'acc-mc': acc_te,
                    'nll-pi_val': nll_val, 'nlpd_val': nlpd_val, 'acc-pi_val': acc_pi_val, 'acc-mc_val': acc_val
                    }
        return results, summary

    dct = {'model_kwargs': model_kwargs, 'callback': callback,
        'process_callback': process_callback, 'name': model_name}
    return key, dct

def initialize_mlp_model_cls(key, features, x, init_var, use_bias=True):
    if isinstance(key, int):
        key = jr.key(key)
    key, subkey = jr.split(key)
    model = MLP(features=features, use_bias=use_bias)
    params = model.init(subkey, x)
    flat_params, unflatten_fn = ravel_pytree(params)
    d = len(flat_params)
    apply_fn = lambda w, x: model.apply(unflatten_fn(w), jnp.atleast_1d(x))
    true_pred_fn = lambda x: model.apply(params, jnp.atleast_1d(x))
    log_likelihood = lambda mean, cov, y: -optax.softmax_cross_entropy(mean, y)
    em_function = apply_fn
    em_linpi_function = lambda w, x: jax.nn.softmax(apply_fn(w, x))
    def ec_function(w, x):
        ps = em_linpi_function(w, x)
        cov = jnp.diag(ps) - jnp.outer(ps, ps) + 1e-5 * jnp.eye(len(ps))
        return jnp.atleast_2d(cov)
    init_kwargs = {
        "init_mean": flat_params,
        "init_cov": init_var,
        "log_likelihood": log_likelihood,
        "emission_mean_function": em_function,
        "em_linpi_function": em_linpi_function,
        "emission_cov_function": ec_function,
        "nparams": nparams,
        "true_pred_fn": true_pred_fn
    }
    return key, init_kwargs

def loss_fn_cls(key, alg, state, em_function, X_val, y_val):
    y_pred_logits = jax.vmap(em_function, (None, 0))(state.mean, X_val)
    negloglikhood = optax.softmax_cross_entropy_with_integer_labels(
        y_pred_logits, y_val
    )
    return jnp.mean(negloglikhood)


def callback_cls(key, alg, state, x, y, em_function, X_te, y_te, X_val, y_val, num_samples=100):
    # Plugin-LL
    ypi_pred_logits_te = jax.vmap(em_function, (None, 0))(state.mean, X_te)
    ll_pi_te = jnp.mean(-optax.softmax_cross_entropy_with_integer_labels(
        ypi_pred_logits_te, y_te
    ))

    ypi_pred_logits_val = jax.vmap(em_function, (None, 0))(state.mean, X_val)
    ll_pi_val = jnp.mean(-optax.softmax_cross_entropy_with_integer_labels(
        ypi_pred_logits_val, y_val
    ))
    
    # Plugin-accuracy
    ypi_preds_te = jnp.argmax(ypi_pred_logits_te, axis=-1)
    acc_pi_te = jnp.mean(ypi_preds_te == y_te)

    ypi_preds_val = jnp.argmax(ypi_pred_logits_val, axis=-1)
    acc_pi_val = jnp.mean(ypi_preds_val == y_val)
    
    # NLPD-LL
    key, subkey = jr.split(key)
    states = alg.sample(subkey, state, num_samples)
    y_pred_logits_te = jnp.mean(jax.vmap(
        jax.vmap(em_function, (None, 0)), (0, None)
    )(states, X_te), axis=0)
    ll_te = jnp.mean(-optax.softmax_cross_entropy_with_integer_labels(
        y_pred_logits_te, y_te
    ))

    key, subkey = jr.split(key)
    states = alg.sample(subkey, state, num_samples)
    y_pred_logits_val = jnp.mean(jax.vmap(
        jax.vmap(em_function, (None, 0)), (0, None)
    )(states, X_val), axis=0)
    ll_val = jnp.mean(-optax.softmax_cross_entropy_with_integer_labels(
        y_pred_logits_val, y_val
    ))
    
    # NLPD-accuracy
    y_preds_te = jnp.argmax(y_pred_logits_te, axis=-1)
    acc_te = jnp.mean(y_preds_te == y_te)

    y_preds_val = jnp.argmax(y_pred_logits_val, axis=-1)
    acc_val = jnp.mean(y_preds_val == y_val)

    return -ll_pi_te, -ll_pi_val, acc_pi_te, acc_pi_val, -ll_te, -ll_val, acc_te, acc_val

