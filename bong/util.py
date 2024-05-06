from functools import partial
from typing import Any, Callable, Sequence
import argparse
from functools import partial
from pathlib import Path
import time
import re
import numpy as np

from flax import linen as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import jax_tqdm
import optax
import optuna
import pandas as pd
import matplotlib.pyplot as plt

from bong.base import RebayesAlgorithm, State
from bong.types import Array, ArrayLike, PRNGKey


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
    def _step(state, t):
        key, subkey = jr.split(jr.fold_in(rng_key, t))
        x, y = X[t], Y[t]
        pred_state = rebayes_algorithm.predict(state)
        output = transform(key, rebayes_algorithm, pred_state, x, y)
        new_state = rebayes_algorithm.update(subkey, pred_state, x, y)
        return new_state, output
    
    if progress_bar:
        _step = jax_tqdm.scan_tqdm(num_timesteps)(_step)
    
    args = jnp.arange(num_timesteps)
    final_state, outputs = jax.lax.scan(_step, init_state, args)
    return final_state, outputs


def tune_init_hyperparam(
    rng_key: PRNGKey,
    rebayes_algorithm_initializer: Any,
    X: ArrayLike,
    Y: ArrayLike,
    loss_fn: Callable,
    hyperparam_name: str,
    n_trials=10,
    minval=-10.0,
    maxval=0.0,
    **init_kwargs,
):
    def _objective(trial):
        init_hp = trial.suggest_float(hyperparam_name, minval, maxval, log=True)
        hp_kwargs = {hyperparam_name: init_hp}
        rebayes_algorithm = rebayes_algorithm_initializer(
            **hp_kwargs,
            **init_kwargs,
        )
        key, subkey = jr.split(rng_key)
        state, _ = run_rebayes_algorithm(
            key, rebayes_algorithm, X, Y,
        )
        eval_loss = loss_fn(subkey, rebayes_algorithm, state)
        return eval_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(_objective, n_trials=n_trials)
    
    best_trial = study.best_trial
    best_params = best_trial.params
    return best_params
    
def run_agents(subkey, agent_queue, data, callback, result_dict={}):
    for agent_name, agent in agent_queue.items():
        print(f"Running {agent_name}...")
        key, subkey = jr.split(subkey)
        t0 = time.perf_counter()
        _, (kldiv, nll, nlpd) = jax.block_until_ready(
            run_rebayes_algorithm(key, agent, data['X_tr'], data['Y_tr'], transform=callback)
        )
        t1 = time.perf_counter()
        result_dict[agent_name] = (t1 - t0, kldiv, nll, nlpd)
        print(f"\tKL-Div: {kldiv[-1]:.4f}, Time: {t1 - t0:.2f}s")
    ntest = len(kldiv)
    result_dict['ntest'] = ntest
    return result_dict


def parse_filename(s):
    # example filename: "fg-l-bong-M10-I10-LR0_01"
    s = s.replace("_", ".")
    import re
    pattern = r"^([a-z-]+)-M(\d+)-I(\d+)-LR(\d*\.?\d+)"
    match = re.match(pattern, s)
    if match:
        return {
            'prefix': match.group(1),
            'M': int(match.group(2)),
            'I': int(match.group(3)),
            'LR': float(match.group(4))
        }
    return {'prefix': None, 'M': None, 'I': None, 'LR': None}

def test_parse_filename():
    strings = ["fg-bong-M10-I1-LR0", "fg-l-bong-M10-I10-LR0_01"]
    for string in strings:
        res = parse_filename(string)
        print(res)

def split_filename_column(df):
    # If filename is fg-bong-M10-I1-LR0_01, we create columns name, M, I, LR with corresponding values

    # Apply the parse function and expand the results into new DataFrame columns
    df_expanded = df['name'].apply(parse_filename).apply(pd.Series)

    # Join the new columns with the original DataFrame
    #df_final = df_expanded.join(df.drop('name', axis=1))
    df_final = df_expanded.join(df)

    # Optionally, rearrange columns to match the desired output format
    #df_final = df_final[['prefix', 'M', 'I', 'LR', 'step', 'kl', 'nll', 'nlpd', 'time']]
    return df_final

def test_split_filename_column():
    # Sample DataFrame
    data = {
        'name': [
            "fg-bong-M10-I1-LR0", "fg-bong-M10-I1-LR0", "fg-bong-M10-I1-LR0",
            "fg-blr-M10-I10-LR0_01", "fg-blr-M10-I10-LR0_01", "fg-blr-M10-I10-LR0_01",
            "fg-blr-M10-I10-LR0_05", "fg-blr-M10-I10-LR0_05", "fg-blr-M10-I10-LR0_05"
        ],
        'step': [0, 1, 2, 0, 1, 2, 0, 1, 2],
        'kl': [4715.3643, np.nan, 4704.921, 4637.003, 4708.9194, 4677.56, 4622.0254, 4707.8594, 4647.589],
        'nll': [0.829247, 0.8394967, 0.84964615, 0.829247, 0.83029026, 0.8313881, 0.829247, 0.8334199, 0.83842194],
        'nlpd': [1.0150638, 1.0070719, 1.00812, 0.98381305, 1.0121765, 1.0324197, 0.98617476, 1.0010813, 0.9961206],
        'time': [
            1.907885749998968, 1.907885749998968, 1.907885749998968,
            2.0240805419743992, 2.0240805419743992, 2.0240805419743992,
            2.021093249961268, 2.021093249961268, 2.021093249961268
        ]
    }
    df = pd.DataFrame(data)
    res = split_filename_column(df)
    print(res)


def extract_nsteps_from_result_dict(result_dict):
    # this breaks for laplace entry, which is a scalar, not a timeseries
    names = list(result_dict.keys())
    r = result_dict[names[0]]
    (tyme, kldiv, nll, nlpd) = r
    T = len(kldiv)
    return T

def convert_result_dict_to_pandas(result_dict):
    result_dict = result_dict.copy()
    #T = extract_nsteps_from_result_dict(result_dict)
    T = result_dict.pop('ntest')
    steps = range(0, T)

    if "laplace" in result_dict:
        laplace = result_dict.pop("laplace")
        (tim, kldiv, nll, nlpd) = laplace
        df  = pd.DataFrame({'name': 'laplace-M0-I0-LR0',  
                                'step': np.array([T]),
                                'kl': np.array(kldiv), 
                                'nll': np.array(nll), 
                                'nlpd': np.array(nlpd),
                                'time': tim,
                                })
        frames = [df]
    else:
        frames = []

    for name, r in result_dict.items():
        df  = pd.DataFrame({'name': name,  
                            'step': steps,
                            'kl': np.array(r[1]), 
                            'nll': np.array(r[2]), 
                            'nlpd': np.array(r[3]),
                            'time': r[0],
                            })
        frames.append(df)
    tbl = pd.concat(frames, ignore_index=True) # ensure each row has unique index
    return tbl


def make_marker(name):
    #https://matplotlib.org/stable/api/markers_api.html
    markers = {'bong': 'o', 'blr': 's', 'bog': 'x', 'bbb': '*'}
    if "bong" in name:
        return markers['bong']
    elif "blr" in name:
        return markers['blr']
    elif "bog" in name:
        return markers['bog']
    elif "bbb" in name:
        return markers['bbb']
    else:
        return 'P;'
    

    
def plot_results(result_dict, curr_path=None, file_prefix='', ttl=''):
    result_dict = result_dict.copy()
    #T = extract_nsteps_from_result_dict(result_dict)
    T = result_dict.pop('ntest')
    # extract subset of points for plotting to avoid cluttered markers
    #ndx = jnp.array(range(0, T, 10)) # decimation of points 
    ndx = round(jnp.linspace(0, T-1, num=min(T,50)))
    # skip first 2 time steps, since it messes up the vertical scale
    ndx = ndx[2:]

    fs = 'small'
    loc = 'lower left'

    # Save KL-divergence, linear scale
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for agent_name, (_, kldiv, _, _) in result_dict.items():
        if jnp.any(jnp.isnan(kldiv)):
            continue
        if agent_name == "laplace":
            ax.axhline(kldiv, color="black", linestyle="--", label=agent_name)
        else:
            ax.plot(ndx, kldiv[ndx], label=agent_name, marker=make_marker(agent_name))
    ax.set_xlabel("number of iteration")
    ax.set_ylabel("KL-divergence")
    #ax.set_yscale("log")
    ax.grid()
    ax.legend()
    ax.legend(loc=loc, prop={'size': fs})
    ax.set_title(ttl)
    if curr_path:
        fname = Path(curr_path, f"{file_prefix}_kl_divergence.pdf")
        fig.savefig(fname, bbox_inches='tight', dpi=300)

    # Save KL-divergence, log scale
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for agent_name, (_, kldiv, _, _) in result_dict.items():
        if jnp.any(jnp.isnan(kldiv)):
            continue
        if agent_name == "laplace":
            ax.axhline(kldiv, color="black", linestyle="--", label=agent_name)
        else:
            ax.plot(ndx, kldiv[ndx], label=agent_name, marker=make_marker(agent_name))
    ax.set_xlabel("number of iteration")
    ax.set_ylabel("KL-divergence")
    ax.set_yscale("log")
    ax.grid()
    ax.legend(loc=loc, prop={'size': fs})
    ax.set_title(ttl)
    if curr_path:
        fname = Path(curr_path, f"{file_prefix}_kl_divergence_logscale.pdf")
        fig.savefig(fname, bbox_inches='tight', dpi=300)
    
    # Save NLL
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for agent_name, (_, _, nll, _) in result_dict.items():
        if jnp.any(jnp.isnan(nll)):
            continue
        if agent_name == "laplace":
            ax.axhline(nll, color="black", linestyle="--", label=agent_name)
        else:
            ax.plot(ndx, nll[ndx], label=agent_name, marker=make_marker(agent_name))
    ax.set_xlabel("number of iteration")
    ax.set_ylabel("NLL (plugin)")
    ax.grid()
    ax.legend(loc=loc, prop={'size': fs})
    ax.set_title(ttl)
    if curr_path:
        fname = Path(curr_path, f"{file_prefix}_plugin_nll.pdf")
        fig.savefig(fname, bbox_inches='tight', dpi=300)

      # Save NLL
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for agent_name, (_, _, nll, _) in result_dict.items():
        if jnp.any(jnp.isnan(nll)):
            continue
        if agent_name == "laplace":
            ax.axhline(nll, color="black", linestyle="--", label=agent_name)
        else:
            ax.plot(ndx, nll[ndx], label=agent_name, marker=make_marker(agent_name))
    ax.set_xlabel("number of iteration")
    ax.set_ylabel("NLL (plugin)")
    ax.set_yscale("log")
    ax.grid()
    ax.legend(loc=loc, prop={'size': fs})
    ax.set_title(ttl)
    if curr_path:
        fname = Path(curr_path, f"{file_prefix}_plugin_nll_logscale.pdf")
        fig.savefig(fname, bbox_inches='tight', dpi=300)
    
    # Save NLPD
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for agent_name, (_, _, _, nlpd) in result_dict.items():
        if jnp.any(jnp.isnan(nlpd)):
            continue
        if agent_name == "laplace":
            ax.axhline(nlpd, color="black", linestyle="--", label=agent_name)
        else:
            ax.plot(ndx, nlpd[ndx], label=agent_name, marker=make_marker(agent_name))
    ax.set_xlabel("number of iteration")
    ax.set_ylabel("NLPD (MC)")
    ax.grid()
    ax.legend(loc=loc, prop={'size': fs})
    ax.set_title(ttl)
    if curr_path:
        fname = Path(curr_path, f"{file_prefix}_mc_nlpd.pdf")
        fig.savefig(fname, bbox_inches='tight', dpi=300)

    # Save runtime
    fig, ax = plt.subplots()
    for agent_name, (runtime, _, _, _) in result_dict.items():
        ax.bar(agent_name, runtime)
    ax.set_ylabel("runtime (s)")
    plt.setp(ax.get_xticklabels(), rotation=30)
    if curr_path:
        fname = Path(curr_path, f"{file_prefix}_runtime.pdf")
        fig.savefig(fname, bbox_inches='tight', dpi=300)
    #plt.close('all')