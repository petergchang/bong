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
from bong.agents import AGENT_DICT, AGENT_NAMES

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
    key = jr.PRNGKey(args.key)
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
    name = f'linreg-dim{args.data_dim}-key{args.key}'
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
    
    return init_kwargs, callback

def run_agent(key, agent, data, callback):
    print(f"Running {agent.name}...")
    t0 = time.perf_counter()
    _, (kldiv, nll, nlpd) = jax.block_until_ready(
        run_rebayes_algorithm(key, agent, data['X_tr'], data['Y_tr'], transform=callback)
    )
    t1 = time.perf_counter()
    print(f"\tKL-Div: {kldiv[-1]:.4f}, Time: {t1 - t0:.2f}s")
    ntest = len(kldiv)
    results = {
        'agent_name': agent.name,
        'dataset_name': data['name'],
        'time': t1 - t0, 
        'kl': kldiv, 
        'nll': nll,
        'nlpd': nlpd, 
        #'ntest': ntest
             }
    return results


def make_results(args):
    if args.dataset == "linreg":
        data, subkey = make_data_linreg(args)
        init_kwargs, callback = init_linreg(args, data)
    else:
        raise Exception(f'unrecognized dataset {args.dataset}')

    constructor = AGENT_DICT[args.agent]['constructor']
    curr_agent = constructor(
                        **init_kwargs,
                        learning_rate = args.lr,
                        num_samples = args.nsample,
                        num_iter = args.niter,
                        linplugin = args.linplugin,
                        empirical_fisher = args.ef
                    )
    results = run_agent(subkey, curr_agent, data, callback)
    df = pd.DataFrame(results)
    return df

def make_dummy_results(args):
    # dummy work
    N = 1000
    key = jr.PRNGKey(0)
    M = jr.normal(key, (N,N))
    M = M * M

    # Save dummy results in dataframe
    T = 10
    steps = jnp.arange(0, T)
    kl = steps * args.lr 
    nll = steps * args.niter
    nlpd = steps * args.nsample
    df  = pd.DataFrame({'time': 0, 
                        'step': steps,
                        'kl': kl,
                        'nll': nll,
                        'nlpd': nlpd
    })
    return df

def extract_args_dict(args, parser):
    args_dict = {action.dest:  getattr(args, action.dest, None) for action in parser._actions}
    args_dict.pop('help')
    return args_dict

def main(args, args_dict):
    print(args)
    results_path = Path(args.dir)
    results_path.mkdir(parents=True, exist_ok=True)

    fname = Path(results_path, f"{args.filename}args.json")
    print("Saving to", fname)
    with open(fname, 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)

    df = make_results(args)

    fname = Path(results_path, f"{args.filename}results.csv")
    print("Saving to", fname)
    df.to_csv(fname, index=False) #, na_rep="NAN", mode="w")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data parameters
    parser.add_argument("--dataset", type=str, default="linreg")
    parser.add_argument("--key", type=int, default=0)
    parser.add_argument("--ntrain", type=int, default=500)
    parser.add_argument("--nval", type=int, default=500)
    parser.add_argument("--ntest", type=int, default=500)
    parser.add_argument("--data_dim", type=int, default=10)
    parser.add_argument("--emission_noise", type=float, default=1.0)
    
    # Model parameters
    parser.add_argument("--agent", type=str, default="fg-bong", choices=AGENT_NAMES)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--niter", type=int, default=10) 
    parser.add_argument("--nsample", type=int, default=10) 
    parser.add_argument("--ef", type=bool, default=True)
    parser.add_argument("--linplugin", type=bool, default=False)

    # results
    parser.add_argument("--dir", type=str, default="", help="directory to store results") 
    parser.add_argument("--filename", type=str, default="", help="filename prefix")
    
    args = parser.parse_args()
    args_dict = extract_args_dict(args, parser)
    main(args, args_dict)

'''
python main.py   --agent bong-fc --lr 0.01 --dir results
'''
