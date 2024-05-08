import argparse
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

#from bong.settings import linreg_path
from bong.src import bbb, blr, bog, bong, experiment_utils
from bong.util import run_rebayes_algorithm, tune_init_hyperparam, run_agents
from bong.util import plot_results, convert_result_dict_to_pandas, split_filename_column

from bong.agents import AGENT_TYPES, LR_AGENT_TYPES, BONG_DICT

import os
cwd = Path(os.getcwd())
#root = cwd.parent.parent
root = cwd



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


def gaussian_kl_div(mu1, sigma1, mu2, sigma2):
    d = mu1.shape[0]
    _, ld1 = jnp.linalg.slogdet(sigma1)
    _, ld2 = jnp.linalg.slogdet(sigma2)
    result = ld2 - ld1 - d
    result += jnp.trace(jnp.linalg.solve(sigma2, sigma1))
    result += (mu2 - mu1).T @ jnp.linalg.solve(sigma2, mu2 - mu1)
    return 0.5 * result

def make_data(args):
     # Generate dataset
    key = jr.PRNGKey(args.key)
    key1, key2, key3, subkey = jr.split(key, 4)
    N, d, noise_std = args.num_examples, args.param_dim, args.emission_noise
    X_tr, Y_tr, theta = generate_linreg_dataset(
        key1, N, d, noise_std=noise_std
    )
    X_val, Y_val, _ = generate_linreg_dataset(
        key2, N, d, noise_std=noise_std, theta=theta
    )
    X_te, Y_te, _ = generate_linreg_dataset(
        key3, N, d, noise_std=noise_std, theta=theta
    )
    data = {'X_tr': X_tr, 'Y_tr': Y_tr, 'X_val': X_val, 'Y_val': Y_val, 'X_te': X_te, 'Y_te': Y_te}
    return data, subkey

def compute_prior_post(args, data):
    # Compute (batch) true posterior
    noise_std = args.emission_noise
    d = args.param_dim
    mu0, cov0 = jnp.ones(d), jnp.eye(d) # Prior moments
    inv_cov0 = jnp.linalg.inv(cov0)
    cov_post = jnp.linalg.inv(inv_cov0 + data['X_tr'].T @ data['X_tr'] / noise_std**2)
    mu_post = cov_post @ (inv_cov0 @ mu0 + data['X_tr'].T @ data['Y_tr'] / noise_std**2)
    post = {'mu': mu_post, 'cov': cov_post}
    prior = {'mu': mu0, 'cov': cov0}
    return prior, post

def init(args, data):
    prior, post = compute_prior_post(args, data)
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

def make_agent_queue(subkey, args, init_kwargs, tune_kl_loss_fn, X_tr, Y_tr):
    agent_queue = {}
    for agent in args.agents:
        if (agent in LR_AGENT_TYPES) and args.tune_learning_rate: 
            for n_sample in args.num_samples:
                for n_iter in args.num_iter:
                    key, subkey = jr.split(subkey)
                    curr_initializer = lambda **kwargs: BONG_DICT[agent](
                        **kwargs,
                        num_iter = n_iter, 
                    )
                    try:
                        best_lr = tune_init_hyperparam(
                            key, curr_initializer, X_tr, Y_tr,
                            tune_kl_loss_fn, "learning_rate", minval=1e-5,
                            maxval=1.0, n_trials=10, **init_kwargs
                        )["learning_rate"]
                    except:
                        best_lr = 1e-2
                    curr_agent = BONG_DICT[agent](
                        learning_rate=best_lr,
                        **init_kwargs,
                        num_samples = n_sample,
                        num_iter = n_iter,
                    )
                    best_lr_str = f"{round(best_lr,4)}".replace('.', '_')
                    agent_queue[f"{agent}-M{n_sample}-I{n_iter}-LR{best_lr_str}-tuned"] = curr_agent
        elif (agent in LR_AGENT_TYPES) and ~args.tune_learning_rate: 
            for n_sample in args.num_samples:
                for n_iter in args.num_iter:
                    for lr in args.learning_rate:
                        lr_str = f"{round(lr,4)}".replace('.', '_')
                        name = f"{agent}-M{n_sample}-I{n_iter}-LR{lr_str}"
                        curr_agent = BONG_DICT[agent](
                            **init_kwargs,
                            learning_rate=lr,
                            num_samples=n_sample,
                            num_iter = n_iter,
                        )
                        agent_queue[name] = curr_agent
        elif "-l-" in agent: # Linearized-BONG (no hparams!)
            curr_agent = BONG_DICT[agent](
                **init_kwargs,
                linplugin=True,
            )
            agent_queue[f"{agent}-M{0}-I{1}-LR{0}"] = curr_agent
        else: # MC-BONG
            for n_sample in args.num_samples:
                curr_agent = BONG_DICT[agent](
                    **init_kwargs,
                    num_samples=n_sample,
                )
                agent_queue[f"{agent}-M{n_sample}-I{1}-LR{0}"] = curr_agent
    return agent_queue, subkey

def debug(args):
    print('DEBUG MODE')
    data, subkey = make_data(args)
    init_kwargs, callback = init(args, data)
    if 1:
        agent = bog.fg_bog(
                        learning_rate = 0.1,
                        num_samples = 100,
                    num_iter = 100,
                        **init_kwargs,
    )
    else:
        agent = bog.make_fg_bog(
                        learning_rate = 0.1,
                        num_samples = 100,
                    num_iter = 100,
                        **init_kwargs,
        )

    print(agent)


def main(args):
    data, subkey = make_data(args)
    init_kwargs, callback = init(args, data)
    if args.debug:
        debug(args)
        return

    prior, post = compute_prior_post(args, data)
    def tune_kl_loss_fn(key, alg, state):
        return gaussian_kl_div(post['mu'], post['cov'], state.mean, state.cov)
      
 
    agent_queue, subkey = make_agent_queue(subkey, args, init_kwargs, tune_kl_loss_fn, data['X_tr'], data['Y_tr'])
    result_dict = run_agents(subkey, agent_queue, data, callback)
  
    
    curr_path = Path(root, "results")
    if args.filename == "":
        filename_prefix =  f"linreg_dim{args.param_dim}"
    else:
        filename_prefix = args.filename
    curr_path.mkdir(parents=True, exist_ok=True)

    fname = Path(curr_path, f"{filename_prefix}.csv")
    print("Saving results to", fname)
    df = convert_result_dict_to_pandas(result_dict)
    df.to_csv(fname, index=False, na_rep="NAN", mode="w")
    
    fname = Path(curr_path, f"{filename_prefix}_parsed.csv")
    df = split_filename_column(df)
    df.to_csv(fname, index=False, na_rep="NAN", mode="w")

    plot_results(result_dict, curr_path, filename_prefix, ttl=filename_prefix)
    

'''
python  experiments/linreg.py  --agents fg-bong fg-bog --param_dim 10 --filename foo \
--num_samples 10 --num_iter 10  --learning_rate 0.05 
'''

   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Data parameters
    parser.add_argument("--num_examples", type=int, default=500)
    parser.add_argument("--param_dim", type=int, default=10)
    parser.add_argument("--key", type=int, default=0)
    parser.add_argument("--emission_noise", type=float, default=1.0)
    
    # Model parameters
    parser.add_argument("--agents", type=str, nargs="+",
                        default=["fg-bong"], choices=AGENT_TYPES)
    parser.add_argument("--num_samples", type=int, nargs="+", 
                        default=[10,])
    
    parser.add_argument("--learning_rate", type=float, nargs="+", 
                    default=[0.005, 0.01, 0.05])
    parser.add_argument("--num_iter", type=int, nargs="+", 
                    default=[10])
    parser.add_argument("--tune_learning_rate", type=bool, default=False)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--filename", type=str, default="")
    
    args = parser.parse_args()
    main(args)