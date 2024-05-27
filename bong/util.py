from functools import partial
from typing import Any, Callable, Sequence
import argparse
from functools import partial
from pathlib import Path
import time
import re
import numpy as np
import datetime
import pytz
import os

from flax import linen as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import jax_tqdm
import optax
import optuna
import pandas as pd
import matplotlib.pyplot as plt
from jax.flatten_util import ravel_pytree
import subprocess


from bong.base import RebayesAlgorithm, State
from bong.types import Array, ArrayLike, PRNGKey


def jax_has_gpu():
    #https://github.com/google/jax/issues/971
    try:
        _ = jax.device_put(jax.numpy.ones(1), device=jax.devices('gpu')[0])
        return True
    except:
        return False


def get_gpu_name():
    if not jax_has_gpu():
        return 'None'
    # Run the nvidia-smi command
    result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True)
    name = result.stdout.strip()
    return name


_vec_pinv = lambda v: jnp.where(v != 0, 1/jnp.array(v), 0) # Vector pseudo-inverse

def move_df_col(df, colname, loc=0):
    cols = df.columns.tolist()
    cols.insert(loc, cols.pop(cols.index(colname)))
    df_reordered = df[cols]
    return df_reordered
    


def delete_generated_txt_files(directory):
    try:
        # Iterate over each file in the specified directory
        for filename in os.listdir(directory):
            # Check if the file ends with '.txt' and starts with 'generated'
            if filename.endswith('.txt') and filename.startswith('created-'):
                # Construct full file path
                full_path = os.path.join(directory, filename)
                # Delete the file
                os.remove(full_path)
                # Optionally print a confirmation
                print(f"Deleted: {full_path}")
    except FileNotFoundError:
        print(f"The directory {directory} does not exist.")
    except PermissionError:
        print(f"Permission denied: cannot delete files in {directory}.")
    except Exception as e:
        print(f"An error occurred: {e}")


def make_file_with_timestamp(dir):
    delete_generated_txt_files(dir)

    now_utc = datetime.datetime.now(pytz.utc)
    # Convert the current time to PST
    pst_timezone = pytz.timezone('US/Pacific')
    now_pst = now_utc.astimezone(pst_timezone)
    #datetime_string = now_pst.strftime("%Y-%m-%d:%H-%M-%S")
    datetime_string = now_pst.strftime("%H:%M-%Y-%m-%d")

    # Create an empty file with the current date-time as its name
    filename = f"{dir}/created-{datetime_string}.txt"
    with open(filename, 'w') as file:
        pass  # This creates an empty file

# https://matplotlib.org/stable/gallery/lines_bars_and_markers/marker_reference.html
marker_types = {
    'point': '.',
    'pixel': ',',
    'circle': 'o',
    'triangle_down': 'v',
    'triangle_up': '^',
    'triangle_left': '<',
    'triangle_right': '>',
    'tri_down': '1',
    'tri_up': '2',
    'tri_left': '3',
    'tri_right': '4',
    'square': 's',
    'pentagon': 'p',
    'star': '*',
    'hexagon1': 'h',
    'hexagon2': 'H',
    'plus': '+',
    'x': 'x',
    'diamond': 'D',
    'thin_diamond': 'd',
    'vline': '|',
    'hline': '_'
}

def make_plot_params(algo, ef, lin):
    if lin:
        markers = {'bong': 'v', 'blr': '*', 'bbb': 'X', 'bog': 's'}
    else:
        markers = {'bong': '^', 'blr': 'd', 'bbb': 'P', 'bog': 'o'}
    marker = markers[algo]
    if (ef==0) and (lin==0): # sampled Hessians
        linestyle = ':'
    else:
        linestyle = '-' 
    return {
            'linestyle': linestyle,
            'linewidth': 2,
            'marker': marker,
            'markersize': 10
            }



def add_jitter(x, jitter_amount=0.1):
    return x + np.random.uniform(-jitter_amount, jitter_amount, size=x.shape)


def convolve_smooth(time_series, width=5, mode='valid'): 
    kernel = jnp.ones(width) / width
    smoothed_time_series = jnp.convolve(time_series, kernel, mode=mode)
    return smoothed_time_series


def safestr(lr):
    '''Convert float to string, replacing . with _, so can be used as a filename.
    0.0056 -> 0_0056, 0.00566 -> 0_0057
    '''
    lr = float(lr) # convert from string if necessary
    lr_str = f"{round(lr,4)}".replace('.', '_')
    return lr_str

def unsafestr(lr_str):
    '''Convert string to float, replacing _ with .'''
    lr_str = lr_str.replace('_', '.')
    return float(lr_str)
    

def make_full_name(algo, param, rank, linplugin, ef, nsamples, niter, lr):
    algo_str = algo
    if param == 'dlr':
        param_str = f'dlr{rank}'
    else:
        param_str = param # fc, fc_mom, diag, diag_mom
    if linplugin == 1:
        hess_str = 'LinHess' # LinHess
    elif ef==1:
        hess_str = f'MCEF{nsamples}' # MC-EF
    else:
        hess_str = f'MCHess{nsamples}' # MC-Hess
    if (algo=="bong") or (algo=="bog"):
        iter_str = None
    else:
        iter_str = f'It{niter}'
    if algo=="bong":
        lr_str = None
    else:
       lr_str =  f'LR{safestr(lr)}'
    s = '-'.join([algo_str, param_str, hess_str])
    if (iter_str is not None):
        s = s + "-" + iter_str
    if (lr_str is not None):
        s = s + "-" + lr_str
    #s = f"{algo}-{param}-R{rank}-Lin{linplugin}-EF{ef}-MC{nsamples}-I{niter}-LR{safestr(lr)}"
    return s


def parse_full_name(s):
    parts = s.split('-')
    unk = 99
    algo, param_str, hess_str = parts[0], parts[1], parts[2]
    if param_str[:3] == 'dlr':
        rank = int(param_str[3:])
        param = 'dlr'
    else:
        rank = unk
        param = param_str
    if hess_str == 'LinHess':
        lin = 1
        ef = 0
        nsamples = unk
    elif hess_str[:4] == 'MCEF':
        lin = 0;
        ef = 1
        nsamples = int(hess_str[4:])
    elif hess_str[:6] == 'MCHess':
        lin = 0
        ef = 0
        nsamples = int(hess_str[6:])
    else:
        print('bad hess_str',hess_str)
    if len(parts)==3: # bong
        assert(algo == 'bong')
        niter = unk
        lr = unk
    if len(parts)==4: # LR but not Iter
        assert(algo == 'bog')
        lr_str = parts[3]
        lr = unsafestr(lr_str[2:])
        niter = unk
    elif len(parts)==5:
        niter_str = parts[3]
        niter = int(niter_str[2:])
        lr_str = parts[4]
        lr = unsafestr(lr_str[2:])
        
    return {
        'full_name': s, 
        'algo': algo,
        'param': param,
        'rank': int(rank),
        'lin': int(lin),
        'ef': int(ef),
        'mc': int(nsamples),
        'niter': int(niter),
        'lr': lr,
        }

    

def make_full_name_old(algo, param, rank, linplugin, ef, nsamples, niter, lr):
    algo_str = algo
    if param == 'dlr':
        param_str = f'dlr{rank}'
    else:
        param_str = param # fc, fc_mom, diag, diag_mom
    if linplugin == 1:
        hess_str = 'Lin' # LinHess
    elif ef==1:
        hess_str = f'EF{nsamples}' # MC-EF
    else:
        hess_str = f'MC{nsamples}' # MC-Hess
    if (algo=="bong") or (algo=="bog"):
        iter_str = None
    else:
        iter_str = f'It{niter}'
    if algo=="bong":
        lr_str = None
    else:
       lr_str =  f'LR{safestr(lr)}'
    s = '-'.join([algo_str, param_str, hess_str])
    if (iter_str is not None):
        s = s + "-" + iter_str
    if (lr_str is not None):
        s = s + "-" + lr_str
    #s = f"{algo}-{param}-R{rank}-Lin{linplugin}-EF{ef}-MC{nsamples}-I{niter}-LR{safestr(lr)}"
    return s

def parse_full_name_old(s):
    parts = s.split('-')
    unk = 99
    algo, param_str, hess_str = parts[0], parts[1], parts[2]
    if param_str[:3] == 'dlr':
        rank = int(param_str[3:])
        param = 'dlr'
    else:
        rank = unk
        param = param_str
    if hess_str == 'Lin':
        lin = 1
        ef = 0
        nsamples = unk
    elif hess_str[:2] == 'EF':
        lin = 0;
        ef = 1
        nsamples = int(hess_str[2:])
    elif hess_str[:2] == 'MC':
        lin = 0
        ef = 0
        nsamples = int(hess_str[2:])
    else:
        print('bad hess_str',hess_str)
    if len(parts)==3: # bong
        assert(algo == 'bong')
        niter = unk
        lr = unk
    if len(parts)==4: # LR but not Iter
        assert(algo == 'bog')
        lr_str = parts[3]
        lr = unsafestr(lr_str[2:])
        niter = unk
    elif len(parts)==5:
        niter_str = parts[3]
        niter = int(niter_str[2:])
        lr_str = parts[4]
        lr = unsafestr(lr_str[2:])
        
    return {
        'full_name': s, 
        'algo': algo,
        'param': param,
        'rank': int(rank),
        'lin': int(lin),
        'ef': int(ef),
        'mc': int(nsamples),
        'niter': int(niter),
        'lr': lr,
        }

def find_first_true(arr):
    true_indices = np.where(arr)[0]
    if true_indices.size > 0:
        first_true_index = true_indices[0]
    else:
        first_true_index = None
    return first_true_index

def list_subdirectories(directory):
    return [name for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))]
            
def gaussian_kl_div(mu1, sigma1, mu2, sigma2):
    d = mu1.shape[0]
    _, ld1 = jnp.linalg.slogdet(sigma1)
    _, ld2 = jnp.linalg.slogdet(sigma2)
    result = ld2 - ld1 - d
    result += jnp.trace(jnp.linalg.solve(sigma2, sigma1))
    result += (mu2 - mu1).T @ jnp.linalg.solve(sigma2, mu2 - mu1)
    return 0.5 * result

class MLPold(nn.Module):
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


# https://twitter.com/DHolzmueller/status/1787053526453023011
#https://jmlr.org/papers/volume23/20-830/20-830.pdf
# Recommends sampling first layer bias terms from U(-sqrt(6), 0)
# to avoid all the kinks (for relu network) being at the origin

def make_bias_initializer(method, minval=-jnp.sqrt(6), maxval=0):
    #https://flax.readthedocs.io/en/latest/api_reference/flax.linen/initializers.html
    def custom_uniform(key, shape, dtype=jnp.float32):
        return jnp.array(jax.random.uniform(key, shape, dtype=dtype, minval=minval, maxval=maxval))

    if method == 'zero':
        bias_init = nn.initializers.constant(0)  
    elif method == 'uniform':
        bias_init = custom_uniform
    else:
        raise Exception(f'unknown bias init {method}')
    return bias_init



class MLP(nn.Module):
    #https://flax.readthedocs.io/en/latest/api_reference/flax.linen/layers.html
    features: Sequence[int]
    activation: nn.Module = nn.gelu
    use_bias: bool = True
    use_bias_first_layer: bool = True
    bias_init_fn: nn.initializers = make_bias_initializer('uniform')
    bias_init_fn_first_layer: nn.initializers = make_bias_initializer('uniform')

    @nn.compact
    def __call__(self, x):
        x = x.ravel()
        if len(self.features) == 1: # linear model
            x = nn.Dense(self.features[-1], use_bias=self.use_bias_first_layer, bias_init=self.bias_init_fn_first_layer)(x)
        else:
            x = self.activation(nn.Dense(self.features[0], use_bias=self.use_bias_first_layer, bias_init=self.bias_init_fn_first_layer)(x))
            for feat in self.features[1:-1]:
                x = self.activation(nn.Dense(feat, use_bias=self.use_bias, bias_init=self.bias_init_fn)(x))
            x = nn.Dense(self.features[-1], use_bias=self.use_bias, bias_init=self.bias_init_fn)(x)
        return x


class CNN(nn.Module):
    input_dim: Sequence[int]
    output_dim: int
    activation: nn.Module = nn.relu
    
    @nn.compact
    def __call__(self, x):
        x = x.reshape(self.input_dim)
        x = nn.Conv(features=16, kernel_size=(5, 5))(x)
        x = self.activation(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=16, kernel_size=(5, 5))(x)
        x = self.activation(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=64)(x)
        x = self.activation(x)
        x = nn.Dense(features=self.output_dim)(x).ravel()
        
        return x


def hess_diag_approx(
    rng_key: PRNGKey,
    fn: Callable,
    param: ArrayLike,
    num_samples: int = 100,
) -> Array:
    """Approximate the diagonal of the Hessian of a function using the
    Hutchinson's method 
        ref: equation (9) of https://arxiv.org/pdf/2006.00719.pdf

    Args:
        rng_key: JAX PRNG Key.
        fn: Function to compute the Hessian of.
        param: Parameters to compute the Hessian at.
        num_samples: Number of samples to use for the approximation.

    Returns:
        Approximate diagonal of the Hessian.
    """
    def _hess_diag(z):
        return z * jax.grad(lambda p: jax.grad(fn)(p) @ z)(param)
    zs = jr.rademacher(rng_key, (num_samples, len(param)))
    return jnp.mean(jax.vmap(_hess_diag)(zs), axis=0)


def fast_svd(
    M: ArrayLike,
) -> Array:
    """Singular value decomposition.

    Args:
        M (m, n): Matrix to decompose.

    Returns:
        U (m, k): Left singular vectors.
        S (k,): Singular values.
    """
    U, S, _ = jnp.linalg.svd(M.T @ M, full_matrices = False, hermitian = True)
    U = M @ (U * _vec_pinv(jnp.sqrt(S)))
    S = jnp.sqrt(S)
    return U, S


def nearest_psd_matrix(mat, eps=1e-6):
    eigenvalues, eigenvectors = jnp.linalg.eigh(mat)
    eigenvalues = jnp.maximum(eigenvalues, eps)
    return (eigenvectors @ jnp.diag(eigenvalues)) @ eigenvectors.T


def sample_dlr_single(key, W, diag, temperature=1.0):
    """
    Sample from an MVG with diagonal + low-rank
    covariance matrix. See §4.2.2, Proposition 1 of
    L-RVGA paper
    """
    key_x, key_eps = jax.random.split(key)
    diag_inv = (1 / diag).ravel()
    diag_inv_mod = diag_inv * temperature
    D, d = W.shape
    
    ID = jnp.eye(D)
    Id = jnp.eye(d)
    
    M = Id + jnp.einsum("ji,j,jk->ik", W, diag_inv, W)
    L = jnp.sqrt(temperature) * \
        jnp.linalg.solve(M.T, jnp.einsum("ji,j->ij", W, diag_inv)).T
    
    x = jax.random.normal(key_x, (D,)) * jnp.sqrt(diag_inv_mod)
    eps = jax.random.normal(key_eps, (d,))
    
    x_plus = jnp.einsum("ij,kj,k->i", L, W, x)
    x_plus = x - x_plus + jnp.einsum("ij,j->i", L, eps)
    return x_plus


@partial(jax.jit, static_argnums=(4,))
def sample_dlr(key, W, diag, temperature=1.0, shape=None):
    shape = (1,) if shape is None else shape
    n_elements = np.prod(shape)
    keys = jax.random.split(key, n_elements)
    samples = jax.vmap(
        sample_dlr_single, in_axes=(0, None, None, None)
    )(keys, W, diag, temperature)
    samples = samples.reshape(*shape, -1)
    return samples


def run_rebayes_algorithm(
    rng_key: PRNGKey,
    rebayes_algorithm: RebayesAlgorithm,
    X: ArrayLike,
    Y: ArrayLike,
    init_state: State=None,
    transform=lambda key, alg, state, x, y: state,
    progress_bar: bool=False,
    n_iter: int=None,
    **init_kwargs,
) -> tuple[State, Any]:
    """Run a rebayes algorithm over a sequence of observations.
    
    Args:
        rng_key: JAX PRNG Key.
        rebayes_algorithm: Rebayes algorithm to run.
        X: Sequence of inputs.
        Y: Sequence of outputs.
        init_state: Initial belief state.
        transform: Transform the belief state after each update.
        progress_bar: Whether to display a progress bar.
        n_iter: Number of iterations to run the algorithm for.
    
    Returns:
        Final belief state and extra information.
    """
    num_timesteps = len(X)
    if init_state is None:
        init_state = rebayes_algorithm.init(**init_kwargs)
    
    @jax.jit
    def _step(state, args):
        x,y,subkey = args
        pred_state = rebayes_algorithm.predict(state)
        output = transform(subkey, rebayes_algorithm, pred_state, x, y)
        new_state = rebayes_algorithm.update(subkey, pred_state, x, y)
        return new_state, output
    
    if progress_bar:
        _step = jax_tqdm.scan_tqdm(num_timesteps)(_step)
    
    #steps = jnp.arange(num_timesteps)
    #final_state, outputs = jax.lax.scan(_step, init_state, steps)
    keys = jr.split(rng_key, num_timesteps)
    final_state, outputs = jax.lax.scan(_step, init_state, (X,Y,keys))
    return final_state, outputs


def tune_init_hyperparam(
    rng_key: PRNGKey,
    rebayes_algorithm_initializer: Any,
    X: ArrayLike,
    Y: ArrayLike,
    loss_fn: Callable,
    hyperparam_names: Sequence[str],
    n_trials=10,
    minval=-10.0,
    maxval=0.0,
    **init_kwargs,
):
    def _objective(trial):
        hp_kwargs = {}
        for hyperparam_name in hyperparam_names:
            init_hp = trial.suggest_float(
                hyperparam_name, minval, maxval, log=True
            )
            hp_kwargs[hyperparam_name] = init_hp
        rebayes_algorithm = rebayes_algorithm_initializer(
            **hp_kwargs,
            **init_kwargs,
        )
        key, subkey = jr.split(rng_key)
        X_fh, X_sh = jnp.split(X, 2)
        Y_fh, Y_sh = jnp.split(Y, 2)
        state_fh, _ = run_rebayes_algorithm(
            key, rebayes_algorithm, X_fh, Y_fh,
        )
        eval_loss_fh = loss_fn(subkey, rebayes_algorithm, state_fh)
        
        key, subkey = jr.split(subkey)
        state_sh, _ = run_rebayes_algorithm(
            key, rebayes_algorithm, X_sh, Y_sh, state_fh
        )
        eval_loss_sh = loss_fn(subkey, rebayes_algorithm, state_sh)

        eval_loss = 0.5 * (eval_loss_fh + eval_loss_sh)
        return eval_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(_objective, n_trials=n_trials)
    
    best_trial = study.best_trial
    best_params = best_trial.params
    return best_params
