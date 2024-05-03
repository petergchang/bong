import argparse
from functools import partial
from pathlib import Path
import time

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

from bong.settings import logreg_path, uci_path
from bong.src import bbb, blr, bog, bong, experiment_utils
from bong.util import MLP, run_rebayes_algorithm, tune_init_hyperparam

tfd = tfp.distributions
MVN = tfd.MultivariateNormalTriL


AGENT_TYPES = ["fg-bong", "fg-l-bong", "fg-rep-bong", "fg-rep-l-bong",
               "fg-blr", "fg-bog", "fg-bbb", "fg-rep-bbb"]
LR_AGENT_TYPES = ["fg-blr", "fg-bog", "fg-bbb", "fg-rep-bbb"]
BONG_DICT = {
    "fg-bong": bong.fg_bong,
    "fg-l-bong": bong.fg_bong,
    "fg-rep-bong": bong.fg_reparam_bong,
    "fg-rep-l-bong": bong.fg_reparam_bong,
    "fg-blr": blr.fg_blr,
    "fg-bog": bog.fg_bog,
    "fg-bbb": bbb.fg_bbb,
    "fg-rep-bbb": bbb.fg_reparam_bbb,
}
UCI_DICT = {
    "uci-statlog-shuttle": 148,
    "uci-covertype": 31,
    "uci-adult": 2,
}


def generate_logreg_dataset(
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


def generate_uci_dataset(
    key, dataset_name, n_train, n_test
):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    dataset = fetch_ucirepo(id=UCI_DICT[dataset_name])
    X = jnp.array(dataset.data.features.to_numpy()).astype('float')
    y = dataset.data.targets.to_numpy().ravel()
    y = jax.nn.one_hot(y, y.max()).astype('float')

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


def logreg_posterior(mu, mu0, cov0, X, Y, em_function,
                     nll_fn=optax.sigmoid_binary_cross_entropy):
    def _ll(x, y):
        return -nll_fn(em_function(mu, x), y)
    ll = jnp.sum(jax.vmap(_ll)(X, Y))
    mvn = MVN(loc=mu0, scale_tril=jnp.linalg.cholesky(cov0))
    log_prior = mvn.log_prob(mu)
    return ll + log_prior


def logreg_kl_div(key, mu0, cov0, mu, cov, X, Y, em_function, n_samples=100,
                  nll_fn=optax.sigmoid_binary_cross_entropy):
    d = mu.shape[0]
    mvn = MVN(loc=mu, scale_tril=jnp.linalg.cholesky(cov))
    mu_samples = mvn.sample((n_samples,), seed=key)
    result = -jnp.mean(
        jax.vmap(
            logreg_posterior, (0, None, None, None, None, None)
        )(mu_samples, mu0, cov0, X, Y, nll_fn)
    )
    _, ld = jnp.linalg.slogdet(cov)
    result += -0.5 * ld - d/2 * (1 + jnp.log(2*jnp.pi))
    return result


def main(args):
    # Generate dataset
    key1, key2, key3, subkey = jr.split(jr.PRNGKey(args.key), 4)
    d = args.param_dim
    if args.dataset == "logreg":
        X_tr, Y_tr, mean_dir = generate_logreg_dataset(key1, args.n_train, d)
        X_val, Y_val, _ = generate_logreg_dataset(key2, args.n_test, 
                                                  d, mean_dir=mean_dir)
        X_te, Y_te, _ = generate_logreg_dataset(key3, args.n_test, 
                                                d, mean_dir=mean_dir)
        # Set up Logistic Regression model
        eps = 1e-5
        sigmoid_fn = lambda w, x: jnp.clip(jax.nn.sigmoid(w @ x), eps, 1-eps)
        log_likelihood = lambda mean, cov, y: \
            -optax.sigmoid_binary_cross_entropy(mean, y)
        em_function = lambda w, x: w @ x
        em_linpi_function = lambda w, x: sigmoid_fn(w, x)
        def ec_function(w, x):
            sigmoid = sigmoid_fn(w, x)
            return jnp.atleast_2d(sigmoid * (1 - sigmoid))        
        # Prior
        mu0 = jnp.ones(d)
        cov0 = args.init_var * jnp.eye(d)
        nll_fn = optax.sigmoid_binary_cross_entropy
        result_path = logreg_path
        def tune_kl_loss_fn(key, alg, state):
            return logreg_kl_div(key, mu0, cov0, state.mean, state.cov,
                                 X_val, Y_val, 100, nll_fn)
    else:
        X_tr, Y_tr, X_te, Y_te = generate_uci_dataset(
            key1, args.dataset, args.n_train, args.n_test
        )
        # Set up multinomial classification model
        *_, n_classes = Y_tr.shape
        model = MLP(features=[50, n_classes,])
        params = model.init(key2, X_tr[0])
        mu0, unflatten_fn = ravel_pytree(params)
        apply_fn = lambda w, x: model.apply(unflatten_fn(w), jnp.atleast_1d(x))
        print(f"num params: {len(mu0)}")
        cov0 = 5e-3*jnp.eye(len(mu0))
        log_likelihood = lambda mean, cov, y: \
            -optax.softmax_cross_entropy(mean, y)
        em_function = apply_fn
        em_linpi_function = lambda w, x: jax.nn.softmax(apply_fn(w, x))
        def ec_function(w, x):
            ps = em_linpi_function(w, x)
            cov = jnp.diag(ps) - jnp.outer(ps, ps) + 1e-5 * jnp.eye(len(ps))
            return jnp.atleast_2d(cov)
        nll_fn = optax.softmax_cross_entropy
        result_path = Path(uci_path, args.dataset)
    # Plugin-NLL
    def _nll(curr_mean, xcb, ycb):
        em = em_function(curr_mean, xcb)
        ec = ec_function(curr_mean, xcb)
        return -log_likelihood(em, ec, ycb)
    
    def callback(key, alg, state, x, y, mu0=mu0, cov0=cov0, 
                 X_cb=X_te, Y_cb=Y_te, n_samples=100):
        # KL-div
        kl_div = logreg_kl_div(key, mu0, cov0, state.mean, state.cov,
                               X_tr, Y_tr,  em_function, n_samples, nll_fn)
        # Plugin-NLL
        nll = jnp.mean(jax.vmap(_nll, (None, 0, 0))(state.mean, X_cb, Y_cb))
        # MC-NLPD
        means = alg.sample(key, state, n_samples)
        nlpds = jnp.mean(jax.vmap(
            jax.vmap(_nll, (None, 0, 0)), (0, None, None)
        )(means, X_cb, Y_cb))
        return kl_div, nll, nlpds
    
    init_linpi_kwargs = {
        "init_mean": mu0,
        "init_cov": cov0,
        "log_likelihood": log_likelihood,
        "emission_cov_function": ec_function,
    }
    init_kwargs = {
        **init_linpi_kwargs,
        "emission_mean_function": em_function,
    }
    agent_queue = {}
    for agent in args.agents:
        # if agent in LR_AGENT_TYPES: # Learning rate TODO: fix this
        #     for n_sample in args.num_samples:
        #         key, subkey = jr.split(subkey)
        #         curr_initializer = lambda **kwargs: BONG_DICT[agent](**kwargs)
        #         try:
        #             best_lr = tune_init_hyperparam(
        #                 key, curr_initializer, X_val, Y_val,
        #                 tune_kl_loss_fn, "learning_rate", minval=1e-4,
        #                 maxval=1.0, **init_kwargs
        #             )["learning_rate"]
        #         except:
        #             best_lr = 1e-2
        #         curr_agent = BONG_DICT[agent](
        #             learning_rate=best_lr,
        #             **init_kwargs,
        #             num_samples=n_sample,
        #         )
        #         agent_queue[f"{agent}-{n_sample}"] = curr_agent
        if "-l-" in agent: # Linearized-BONG
            curr_agent = BONG_DICT[agent](
                **init_linpi_kwargs,
                linplugin=True,
                emission_mean_function=em_linpi_function,
            )
            agent_queue[agent] = curr_agent
        else: # MC-BONG
            for n_sample in args.num_samples:
                curr_agent = BONG_DICT[agent](
                    **init_kwargs,
                    num_samples=n_sample,
                    learning_rate=0.005,
                )
                agent_queue[f"{agent}-{n_sample}"] = curr_agent
    result_dict = {}

    # Run Laplace baseline
    if args.dataset == "logreg":
        print("Running Laplace...")
        t0 = time.perf_counter()
        key, subkey = jr.split(subkey)
        sol_lap = minimize(
            lambda *args: -logreg_posterior(*args), mu0, 
            args=(mu0, cov0, X_tr, Y_tr, em_function), method="BFGS", 
            options={'line_search_maxiter': 1e2, 'gtol': args.laplace_gtol}
        )
        if not sol_lap.success:
            raise ValueError("Laplace failed to converge. Increase tolerance.")
        mu_lap = sol_lap.x
        nll_sum_fn = lambda m, X, Y: \
            jnp.sum(jax.vmap(_nll, (None, 0, 0))(m, X, Y))
        nll_mean_fn = lambda m, X, Y: \
            jnp.mean(jax.vmap(_nll, (None, 0, 0))(m, X, Y))
        cov_lap = jnp.linalg.pinv(jax.hessian(nll_sum_fn)(mu_lap, X_tr, Y_tr))
        kldiv_lap = logreg_kl_div(key, mu0, cov0, mu_lap, cov_lap, X_tr, Y_tr, em_function)
        nll_lap = nll_mean_fn(mu_lap, X_te, Y_te)
        mvn = MVN(loc=mu_lap, scale_tril=jnp.linalg.cholesky(cov_lap))
        means_lap = mvn.sample((100,), seed=subkey)
        nlpd_lap = jnp.mean(
            jax.vmap(nll_mean_fn, (0, None, None))(means_lap, X_te, Y_te)
        )
        t1 = time.perf_counter()
        result_dict["laplace"] = (t1 - t0, kldiv_lap, nll_lap, nlpd_lap)
    
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
    curr_path = Path(result_path, f"dim_{args.param_dim}")
    curr_path.mkdir(parents=True, exist_ok=True)
    print("saving to", curr_path)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for agent_name, (_, kldiv, _, _) in result_dict.items():
        if jnp.any(jnp.isnan(kldiv)):
            continue
        if agent_name == "laplace":
            ax.axhline(kldiv, color="black", linestyle="--", label=agent_name)
        else:
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
        if agent_name == "laplace":
            ax.axhline(nll, color="black", linestyle="--", label=agent_name)
        else:
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
        if agent_name == "laplace":
            ax.axhline(nlpd, color="black", linestyle="--", label=agent_name)
        else:
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
    parser.add_argument("--dataset", type=str, default="logreg", 
                        choices=["logreg", "uci-statlog-shuttle",
                                 "uci-covertype", "uci-adult"])
    parser.add_argument("--n_train", type=int, default=1_000)
    parser.add_argument("--n_test", type=int, default=1_000)
    parser.add_argument("--param_dim", type=int, default=10)
    parser.add_argument("--key", type=int, default=0)
    
    # Model parameters
    parser.add_argument("--agents", type=str, nargs="+",
                        default=["fg-bong"], choices=AGENT_TYPES)
    parser.add_argument("--num_samples", type=int, nargs="+", 
                        default=[100])
    parser.add_argument("--init_var", type=float, default=4.0)
    parser.add_argument("--laplace_gtol", type=float, default=1e-3)
    
    args = parser.parse_args()
    main(args)