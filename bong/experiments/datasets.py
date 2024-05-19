

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
import os


from bong.util import run_rebayes_algorithm, gaussian_kl_div, MLP
from job_utils import make_neuron_str, parse_neuron_str
from bong.src import bbb, blr, bog, bong, experiment_utils
from models import initialize_mlp_model_reg, initialize_mlp_model_cls


tfd = tfp.distributions
MVN = tfd.MultivariateNormalTriL

def add_column_of_ones(A):
    ones_column = jnp.ones((A.shape[0], 1))
    A_with_ones = jnp.hstack((A, ones_column))
    return A_with_ones

def add_ones_to_covariates(data):
    data['X_tr'] = add_column_of_ones(data['X_tr'])
    data['X_val'] = add_column_of_ones(data['X_val'])
    data['X_te'] = add_column_of_ones(data['X_te'])
    return data

def make_dataset(key, args):
    if args.dataset == "reg":
        if args.dgp_type == "lin":
            key, data = make_data_reg_lin(key, args)
        elif args.dgp_type == "mlp":
            key, data = make_data_reg_mlp(key, args)
        else:
            raise Exception(f'Unknown dgp {args.dgp}')
    elif args.dataset == "cls":
        if args.dgp_type == "lin":
            key, data = make_data_cls_lin(key, args)
        elif args.dgp_type == "mlp":
            key, data = make_data_cls_mlp(key. args)
        else:
            raise Exception(f'Unknown dgp {args.dgp}')
    elif args.dataset == "sarcos":
        key, data = get_sarcos_data(key, args)

    return key, data

### SARCOS robot arm data


def get_sarcos_data(key, args):
    # https://gaussianprocess.org/gpml/data/
    # We add column of 1s by default to match linreg results in sarcos_demo.py
    # But if passing to a neural net with a bias term, this is unnecessary
    ntrain, nval, ntest = args.ntrain, args.nval, args.ntest
    print('get sarcos', ntrain, nval, ntest)
    import scipy.io
    cwd = Path(os.getcwd())
    root = cwd
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    #folder = '/teamspace/studios/this_studio/bong/bong/data'
    folder = f'{script_dir}/../data/'

    mat_data = scipy.io.loadmat(f'{folder}/sarcos_inv.mat') # (44484, 28)
    data_train = jnp.array(mat_data['sarcos_inv'])
    max_ntrain = data_train.shape[0] 

    # Shuffle the rows
    key, subkey = jr.split(key)
    perm = jr.permutation(subkey, max_ntrain)
    data_train = data_train[perm]

    if ntrain == 0:
        ntrain = max_ntrain
    else:
        ntrain = min(ntrain, max_ntrain)
    idx_tr = jnp.arange(0, ntrain)
    X_tr = data_train[idx_tr, :21]
    Y_tr = data_train[idx_tr, 21] # column 22

    if (nval == 0) or (ntrain+nval > max_ntrain):
        X_val, Y_val = X_tr, Y_tr
    else:
        idx_val = jnp.arange(ntrain, ntrain+nval)
        X_val = data_train[idx_val, :21]
        Y_val = data_train[idx_val, 21] # column 22

    mat_data = scipy.io.loadmat(f'{folder}/sarcos_inv_test.mat') # (4449, 28)
    data_test = jnp.array(mat_data['sarcos_inv_test'])
    max_ntest = data_test.shape[0] 
    if ntest == 0:
        ntest = max_ntest 
    else:
        ntest = min(ntest, max_ntest)
    idx_te = jnp.arange(0, ntest)
    X_te = data_test[idx_te, :21]
    Y_te = data_test[idx_te, 21] # column 22

    name = 'sarcos'
    # We also return "raw" data, such as X_train, for debugging (see sarcos_demo.py)
    data = {
        'X_tr': X_tr, 'Y_tr': Y_tr, 'X_val': X_val, 'Y_val': Y_val, 'X_te': X_te, 'Y_te': Y_te, 'name': name, 
        'X_train_raw': data_train[:, :21], 'Y_train_raw': data_train[:, 21],
        'X_test_raw': data_test[:, :21], 'Y_test_raw': data_test[:, 21],
    }
    data = standardize_xy_data(data, add_ones=True)
    # Need to set add_ones=True otherwise get the error
    #jax.errors.TracerArrayConversionError: The numpy.ndarray conversion
    # # method __array__() was called on traced array with shape int32[].

    return key, data


def standardize_xy_data(data, add_ones): 
    data = data.copy()
    scaler = preprocessing.StandardScaler().fit(data['X_tr'])
    Xtrain = scaler.transform(data['X_tr'])
    Xval = scaler.transform(data['X_val'])
    Xtest = scaler.transform(data['X_te'])
    if add_ones:
        Xtrain = add_column_of_ones(Xtrain)
        Xval = add_column_of_ones(Xval)
        Xtest = add_column_of_ones(Xtest)
    ytrain, yval, ytest = data['Y_tr'], data['Y_val'], data['Y_te']
    mu_y, v_y = jnp.mean(ytrain), jnp.var(ytrain)
    ytrain, yval, ytest = ytrain - mu_y, yval - mu_y, ytest - mu_y
    data['X_tr'], data['Y_tr'] = Xtrain, ytrain
    data['X_val'], data['Y_val'] = Xval, yval
    data['X_te'], data['Y_te'] = Xtest, ytest
    return data



### LINREG generators

def generate_xdata_mvn(key, N, d, c=1., scale=1):
    key1, key2, key = jr.split(key, 3)
    mean = jnp.zeros(d)
    cov = experiment_utils.generate_covariance_matrix(key1, d, c, scale)
    X = jr.multivariate_normal(key1, mean, cov, (N,))
    return X

def generate_linear_model(key, d):
    key1, key2, key = jr.split(key, 3)
    theta = jr.uniform(key1, (d,), minval=-1., maxval=1.)
    theta = theta / jnp.linalg.norm(theta)
    return theta

def generate_ydata_linreg(key, X, theta, noise_std=1.0):
    key1, key = jr.split(key)
    N, d = X.shape
    Ymean = X @ theta # (N,)
    Y = Ymean  + jr.normal(key1, (N,)) * noise_std
    return Y


def make_data_reg_lin(key, args):
    d, noise_std = args.data_dim, args.emission_noise
    key0, key = jr.split(key)
    theta = generate_linear_model(key0, d)
    name = f'reg-D{args.data_dim}-lin_1'

    key1, key2, key3, key = jr.split(key, 4)
    X_tr = generate_xdata_mvn(key1, args.ntrain, d)
    Y_tr = generate_ydata_linreg(key1, X_tr, theta, noise_std)
    X_val = generate_xdata_mvn(key2, args.nval, d)
    Y_val = generate_ydata_linreg(key2, X_val, theta, noise_std)
    X_te = generate_xdata_mvn(key3, args.ntest, d)
    Y_te = generate_ydata_linreg(key3, X_te, theta, noise_std)


    data = {'X_tr': X_tr, 'Y_tr': Y_tr, 'X_val': X_val, 'Y_val': Y_val, 'X_te': X_te, 'Y_te': Y_te, 'name': name}
    return key, data


##########  MLP generators

def generate_ydata_mlp_reg(key, X, predictor, noise_std=1.0):
    subkey, key = jr.split(key)
    N, d = X.shape
    Ymean = jax.vmap(predictor)(X)  # (N,1) for scalar output
    Y = Ymean + jr.normal(subkey, (N,1)) * noise_std
    Y = Y.flatten() # convert to (N,)
    return Y




def make_data_reg_mlp(key, args):
    neurons_str = args.dgp_str
    name = f'reg-D{args.data_dim}-mlp_{neurons_str}'
    neurons = parse_neuron_str(neurons_str)

    d, noise_std = args.data_dim, args.emission_noise
    key0, key = jr.split(key)
    x = jnp.zeros(d)
    #model = generate_mlp_reg(key1, args.data_dim, args.dgp_neurons)
    key, model= initialize_mlp_model_reg(key0, neurons, x, args.init_var, args.emission_noise)
    predictor = model['true_pred_fn']

    key1, key2, key3, key = jr.split(key, 4)
    X_tr = generate_xdata_mvn(key1, args.ntrain, d)
    Y_tr = generate_ydata_mlp_reg(key1, X_tr, predictor, noise_std)
    X_val = generate_xdata_mvn(key2, args.nval, d)
    Y_val = generate_ydata_mlp_reg(key2, X_val, predictor, noise_std)
    X_te = generate_xdata_mvn(key3, args.ntest, d)
    Y_te = generate_ydata_mlp_reg(key3, X_te, predictor, noise_std)

    data = {'X_tr': X_tr, 'Y_tr': Y_tr, 'X_val': X_val, 'Y_val': Y_val, 'X_te': X_te, 'Y_te': Y_te, 'name': name}
    return key, data

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

def make_data_cls_lin(key, args):
    d = args.data_dim
    name = f'cls-D{args.data_dim}-lin_1'
    key1, key2, key3, key = jr.split(key, 4)
    X_tr, Y_tr, theta = generate_logreg_dataset(key1, args.n_train, d)
    X_val, Y_val, _ = generate_logreg_dataset(key2, args.n_val, d, theta=theta)
    X_te, Y_te, _ = generate_logreg_dataset(key3, args.n_test, d, theta=theta)
    data = {'X_tr': X_tr, 'Y_tr': Y_tr, 'X_val': X_val, 'Y_val': Y_val, 'X_te': X_te, 'Y_te': Y_te, 'name': name}
    return key, data



##########  MLP generators for classification

def generate_ydata_mlp_cls(key, X, predictor):
    subkey, key = jr.split(key)
    N, d = X.shape
    predictor = predictor
    Ymean = jax.vmap(predictor)(X)  # (N,1) for scalar output
    Y = Ymean 
    #Y = Y.flatten() # convert to (N,)
    return Y



def make_data_cls_mlp(key, args):
    neurons_str = args.dgp_str
    name = f'cls-D{args.data_dim}-mlp_{neurons_str}'
    neurons = parse_neuron_str(neurons_str)

    d, noise_std = args.data_dim, args.emission_noise
    key0, key = jr.split(key)
    x = jnp.zeros(d)
    model, key = initialize_mlp_model_cls(key0, neurons, x, args.init_var)
    predictor = model['true_pred_fn']

    key1, key2, key3, key = jr.split(key, 4)
    X_tr = generate_xdata_mvn(key1, args.ntrain, d)
    Y_tr = generate_ydata_mlp_cls(key1, X_tr, predictor)
    X_val = generate_xdata_mvn(key2, args.nval, d)
    Y_val = generate_ydata_mlp_cls(key2, X_val, predictor)
    X_te = generate_xdata_mvn(key3, args.ntest, d)
    Y_te = generate_ydata_mlp_cls(key3, X_te, predictor)

    data = {'X_tr': X_tr, 'Y_tr': Y_tr, 'X_val': X_val, 'Y_val': Y_val, 'X_te': X_te, 'Y_te': Y_te, 'name': name}
    return key, data




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
