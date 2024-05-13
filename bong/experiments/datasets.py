

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
from sklearn import preprocessing
import tensorflow_probability.substrates.jax as tfp
from ucimlrepo import fetch_ucirepo


from bong.util import run_rebayes_algorithm, gaussian_kl_div, MLP
from bong.src import bbb, blr, bog, bong, experiment_utils

tfd = tfp.distributions
MVN = tfd.MultivariateNormalTriL

def make_dataset(args):
    if args.dataset == "reg":
        dgp = args.dgp[0:3] #lin_1, mlp_20_20_1
        if dgp == "lin":
            data = make_data_reg_lin(args)
        elif dgp == "mlp":
            data = make_data_reg_mlp(args)
        else:
            raise Exception(f'Unknown dgp {args.dgp}')
    elif args.dataset == "cls":
        if dgp == "lin":
            data = make_data_cls_lin(args)
        else:
            raise Exception(f'Unknown dgp {args.dgp}')
    return data

### LINREG generators

def generate_xdata_mvn(keyroot, N, d, c=1., scale=1):
    key0, key1, keyroot = jr.split(keyroot, 3)
    mean = jnp.zeros(d)
    cov = experiment_utils.generate_covariance_matrix(key0, d, c, scale)
    X = jr.multivariate_normal(key1, mean, cov, (N,))
    return X

def generate_linear_model(keyroot, d):
    key1, key2, keyroot = jr.split(keyroot, 3)
    theta = jr.uniform(key1, (d,), minval=-1., maxval=1.)
    theta = theta / jnp.linalg.norm(theta)
    return theta

def generate_ydata_linreg(keyroot, X, theta, noise_std=1.0):
    key, keyroot = jr.split(keyroot)
    N, d = X.shape
    Ymean = X @ theta # (N,)
    Y = Ymean  + jr.normal(key, (N,)) * noise_std
    return Y


def make_data_reg_lin(args):
    d, noise_std = args.data_dim, args.emission_noise
    keyroot = jr.PRNGKey(args.data_key)
    key1, keyroot = jr.split(keyroot)
    theta = generate_linear_model(key1, args.data_dim)
    name = f'reg-D{args.data_dim}-lin_1'

    key1, key2, key3, keyroot = jr.split(keyroot, 4)
    X_tr = generate_xdata_mvn(key1, args.ntrain, d)
    Y_tr = generate_ydata_linreg(key1, X_tr, theta, noise_std)
    X_val = generate_xdata_mvn(key2, args.nval, d)
    Y_val = generate_ydata_linreg(key2, X_val, theta, noise_std)
    X_te = generate_xdata_mvn(key3, args.ntest, d)
    Y_te = generate_ydata_linreg(key3, X_te, theta, noise_std)


    data = {'X_tr': X_tr, 'Y_tr': Y_tr, 'X_val': X_val, 'Y_val': Y_val, 'X_te': X_te, 'Y_te': Y_te, 'name': name}
    return data


##########  MLP generators

def generate_ydata_mlpreg(keyroot, X, model, noise_std=1.0):
    key, keyroot = jr.split(keyroot)
    N, d = X.shape
    predictor = model['pred_fn']
    Ymean = jax.vmap(predictor)(X)  # (N,1) for scalar output
    Y = Ymean + jr.normal(key, (N,1)) * noise_std
    return Y


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


def make_data_reg_mlp(args):
    dgp = args.dgp
    neurons = [int(n) for n in dgp[4:].split("_")] # mlp_10_10_1 or mlp_1
    name = f'reg-D{args.data_dim}-{args.dgp}'

    d, noise_std = args.data_dim, args.emission_noise
    keyroot = jr.PRNGKey(args.data_key)
    key1, keyroot = jr.split(keyroot)
    model = generate_mlp(key1, args.data_dim, neurons)

    key1, key2, key3, keyroot = jr.split(keyroot, 4)
    X_tr = generate_xdata_mvn(key1, args.ntrain, d)
    Y_tr = generate_ydata_mlpreg(key1, X_tr, model, noise_std)
    X_val = generate_xdata_mvn(key2, args.nval, d)
    Y_val = generate_ydata_mlpreg(key2, X_val, model, noise_std)
    X_te = generate_xdata_mvn(key3, args.ntest, d)
    Y_te = generate_ydata_mlpreg(key3, X_te, model, noise_std)

    data = {'X_tr': X_tr, 'Y_tr': Y_tr, 'X_val': X_val, 'Y_val': Y_val, 'X_te': X_te, 'Y_te': Y_te, 'name': name}
    return data

### LOGREG generators

def generate_logreg_dataset_from_gmm(
    key, N, d, c=1., scale=1., coef_s=0.2, mean_dir=None
):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    keys = jr.split(key, 5)
    mean_sep = 1/d**coef_s
    if mean_dir is None:
        mean_dir = jr.uniform(keys[0], (d,))
        mean_dir = mean_dir / jnp.linalg.norm(mean_dir)
    mean_zero = mean_dir * mean_sep/2
    mean_one = -mean_dir * mean_sep/2
    cov = experiment_utils.generate_covariance_matrix(keys[1], d, c, scale)
    cov_inv = jnp.linalg.inv(cov)
    theta = cov_inv @ (mean_one - mean_zero)
    X_zero = jr.multivariate_normal(keys[2], mean_zero, cov, (N//2,))
    X_one = jr.multivariate_normal(keys[3], mean_one, cov, (N-N//2,))
    X = jnp.concatenate([X_zero, X_one])
    Y = jnp.concatenate([jnp.zeros(N//2), jnp.ones(N-N//2)])
    # Shuffle
    idx = jr.permutation(keys[4], len(X))
    X, Y = X[idx], Y[idx]
    return X, Y, mean_dir

def logistic(r):
    return 1 / (1 + jnp.exp(-r))

def generate_logreg_dataset(
    key, N, d, c=1., scale=1., theta=None
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
    #Y = X @ theta + jr.normal(keys[3], (N,)) * noise_std
    eps = 1e-5
    probs = jnp.clip(jax.nn.sigmoid(X @ theta), eps, 1-eps)
    Y = jr.bernoulli(keys[3], probs) 
    return X, Y, theta

def make_data_cls_lin(args):
    d = args.data_dim
    name = f'cls-D{args.data_dim}-lin1'
    keyroot = jr.PRNGKey(args.data_key)
    key1, key2, key3, keyroot = jr.split(keyroot, 4)
    X_tr, Y_tr, theta = generate_logreg_dataset(key1, args.n_train, d)
    X_val, Y_val, _ = generate_logreg_dataset(key2, args.n_val, d, theta=theta)
    X_te, Y_te, _ = generate_logreg_dataset(key3, args.n_test, d, theta=theta)
    data = {'X_tr': X_tr, 'Y_tr': Y_tr, 'X_val': X_val, 'Y_val': Y_val, 'X_te': X_te, 'Y_te': Y_te, 'name': name}
    return data


# UCI is unfinished

UCI_DICT = {
    "uci-statlog-shuttle": 148,
    "uci-covertype": 31,
    "uci-adult": 2,
}


def generate_uci_dataset(
    key, dataset_name, n_train, n_test
):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    dataset = fetch_ucirepo(id=UCI_DICT[dataset_name])
    X = jnp.array(dataset.data.features.to_numpy()).astype('float')
    y = dataset.data.targets.to_numpy().ravel()
    y = jax.nn.one_hot(y, y.max()).astype('float')

    print(f"UCI dataset {dataset_name}, nex: {X.shape[0]}, ndim: {X.shape[1]}")
    
    # Shuffle and split
    idx = jr.permutation(key, len(X))
    X, y = X[idx], y[idx]
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_te, y_te = X[n_train:n_train+n_test], y[n_train:n_train+n_test]

    # Preprocess
    scaler = preprocessing.StandardScaler().fit(X_tr)
    X_tr = jnp.array(scaler.transform(X_tr))
    X_te = jnp.array(scaler.transform(X_te))
    return X_tr, y_tr, X_te, y_te
