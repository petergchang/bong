import argparse
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from bong.settings import linreg_path
from bong.src import bong, experiment_utils
from bong.util import run_rebayes_algorithm


AGENT_TYPES = ["fg-bong", "fg-l-bong", "fg-rep-bong", "fg-rep-l-bong"]
BONG_DICT = {
    "fg-bong": bong.fg_bong,
    "fg-l-bong": bong.fg_bong,
    "fg-rep-bong": bong.fg_reparam_bong,
    "fg-rep-l-bong": bong.fg_reparam_bong,
}


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


def main(args):
    # Generate dataset
    key1, key2, subkey = jr.split(jr.PRNGKey(args.key), 3)
    N, d, noise_std = args.num_examples, args.param_dim, args.emission_noise
    X_tr, Y_tr, theta = generate_linreg_dataset(
        key1, N, d, noise_std=noise_std
    )
    X_te, Y_te, _ = generate_linreg_dataset(
        key2, N, d, noise_std=noise_std, theta=theta
    )
    
    # Compute (batch) true posterior
    mu0, cov0 = jnp.ones(d), jnp.eye(d) # Prior moments
    inv_cov0 = jnp.linalg.inv(cov0)
    cov_post = jnp.linalg.inv(inv_cov0 + X_tr.T @ X_tr / noise_std**2)
    mu_post = cov_post @ (inv_cov0 @ mu0 + X_tr.T @ Y_tr / noise_std**2)
    
    # Compute KL divergence, plugin NLL, MC-NLPD with agents
    log_likelihood = lambda mean, cov, y: \
        jax.scipy.stats.norm.logpdf(y, mean, jnp.sqrt(jnp.diag(cov))).sum()
    em_function = lambda w, x: w @ x
    ec_function = lambda w, x: noise_std * jnp.eye(1)
    def callback(key, alg, state, x, y, X_cb=X_te, Y_cb=Y_te, n_samples=100):
        # KL-div
        kl_div = gaussian_kl_div(mu_post, cov_post, state.mean, state.cov)
        # Plugin-NLL
        def _nll(curr_mean, xcb, ycb):
            em = em_function(curr_mean, xcb)
            ec = ec_function(curr_mean, xcb)
            return -log_likelihood(em, ec, ycb)
        nll = jnp.mean(jax.vmap(_nll, (None, 0, 0))(state.mean, X_cb, Y_cb))
        # MC-NLPD
        means = alg.sample(key, state, n_samples)
        nlpd = jnp.mean(jax.vmap(
            jax.vmap(_nll, (None, 0, 0)), (0, None, None)
        )(means, X_cb, Y_cb))
        return kl_div, nll, nlpd
    
    init_kwargs = {
        "init_mean": mu0,
        "init_cov": cov0,
        "log_likelihood": log_likelihood,
        "emission_mean_function": em_function,
        "emission_cov_function": ec_function,
    }
    agent_queue = {}
    for agent in args.agents:
        if "-l-" in agent: # Linearized-BONG
            curr_agent = BONG_DICT[agent](
                **init_kwargs,
                linplugin=True,
            )
            agent_queue[agent] = curr_agent
        else: # MC-BONG
            for n_sample in args.num_samples:
                curr_agent = BONG_DICT[agent](
                    **init_kwargs,
                    num_samples=n_sample,
                )
                agent_queue[f"{agent}-{n_sample}"] = curr_agent
    result_dict = {}
    for agent_name, agent in agent_queue.items():
        print(f"Running {agent_name}...")
        key, subkey = jr.split(subkey)
        t0 = time.perf_counter()
        _, (kldiv, nll, nlpd) = jax.block_until_ready(
            run_rebayes_algorithm(key, agent, X_tr, Y_tr, transform=callback)
        )
        t1 = time.perf_counter()
        result_dict[agent_name] = (t1 - t0, kldiv, nll, nlpd)
        print(f"\tKL-Div: {kldiv[-1]:.4f}, Time: {t1 - t0:.2f}s")
        
    # Save KL-divergence
    curr_path = Path(linreg_path, f"dim_{args.param_dim}")
    print("Saving to", curr_path)
    curr_path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for agent_name, (_, kldiv, _, _) in result_dict.items():
        if jnp.any(jnp.isnan(kldiv)):
            continue
        ax.plot(kldiv, label=agent_name)
    ax.set_xlabel("number of iteration")
    ax.set_ylabel("KL-divergence")
    ax.set_yscale("log")
    ax.grid()
    ax.legend()
    fig.savefig(
        Path(curr_path, f"kl_divergence.pdf"), bbox_inches='tight', dpi=300
    )
    
    # Save NLL
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for agent_name, (_, _, nll, _) in result_dict.items():
        if jnp.any(jnp.isnan(nll)):
            continue
        ax.plot(nll, label=agent_name)
    ax.set_xlabel("number of iteration")
    ax.set_ylabel("NLL (plugin)")
    ax.grid()
    ax.legend()
    fig.savefig(
        Path(curr_path, f"plugin_nll.pdf"), bbox_inches='tight', dpi=300
    )
    
    # Save NLPD
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for agent_name, (_, _, _, nlpd) in result_dict.items():
        if jnp.any(jnp.isnan(nlpd)):
            continue
        ax.plot(nlpd, label=agent_name)
    ax.set_xlabel("number of iteration")
    ax.set_ylabel("NLPD (MC)")
    ax.grid()
    ax.legend()
    fig.savefig(
        Path(curr_path, f"mc_nlpd.pdf"), bbox_inches='tight', dpi=300
    )

    # Save runtime
    fig, ax = plt.subplots()
    for agent_name, (runtime, _, _, _) in result_dict.items():
        ax.bar(agent_name, runtime)
    ax.set_ylabel("runtime (s)")
    plt.setp(ax.get_xticklabels(), rotation=30)
    fig.savefig(
        Path(curr_path, f"runtime.pdf"), bbox_inches='tight', dpi=300
    )
    plt.close('all')
    

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
                        default=[1, 10, 100])
    
    args = parser.parse_args()
    main(args)