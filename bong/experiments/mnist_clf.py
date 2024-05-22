import argparse
from functools import partial
import json
from pathlib import Path
import pickle
import time

import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax
import tqdm

from bong.agents import DIAG_BONG_DICT, DLR_BONG_DICT
from bong.settings import hparam_mnist_path, mnist_path
import bong.src.dataloaders as dataloaders
from bong.util import MLP, CNN, run_rebayes_algorithm, tune_init_hyperparam


BONG_DICT = {**DIAG_BONG_DICT, **DLR_BONG_DICT}


def _farray(x):
    x = jnp.array(x)
    x = jnp.where(
        x < -1e1, jnp.nan, jnp.where(
            x > 1e1, jnp.nan, x
        )
    )
    return x


def load_mnist_dataset(n_train, n_test):
    dataset_fn, _ = dataloaders.generate_stationary_experiment(
        n_train, ntest=n_test
    ).values()
    dataset = dataset_fn()
    X_tr, Y_tr = dataset['train']
    X_val, Y_val = dataset['val']
    X_te, Y_te = dataset['test']
    return X_tr, Y_tr, X_val, Y_val, X_te, Y_te


def loss_fn(key, alg, state, em_function, X_val, y_val):
    y_pred_logits = jax.vmap(em_function, (None, 0))(state.mean, X_val)
    negloglikhood = optax.softmax_cross_entropy_with_integer_labels(
        y_pred_logits, y_val
    )
    return jnp.mean(negloglikhood)


def callback_fn(key, alg, state, x, y, em_function, X_cb, y_cb, num_samples=10):
    # Plugin-NLL
    ypi_pred_logits = jax.vmap(em_function, (None, 0))(state.mean, X_cb)
    nll_pi = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
        ypi_pred_logits, y_cb
    ))
    
    # Plugin-misclassification
    ypi_preds = jnp.argmax(ypi_pred_logits, axis=-1)
    miscl_pi = jnp.mean(ypi_preds != y_cb)
    
    # NLPD-NLL
    states = alg.sample(key, state, num_samples)
    y_pred_logits = jnp.mean(jax.vmap(
        jax.vmap(em_function, (None, 0)), (0, None)
    )(states, X_cb), axis=0)
    nll = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
        y_pred_logits, y_cb
    ))
    
    # NLPD-misclassification
    y_preds = jnp.argmax(y_pred_logits, axis=-1)
    miscl = jnp.mean(y_preds != y_cb)
    
    # Linearized-NLL
    def _linearized_apply(w, x):
        H = jax.jacrev(lambda ww, xx: em_function(ww, xx).squeeze())(state.mean, x)
        return (em_function(state.mean, x) + H @ (w - state.mean)).squeeze()
    ylin_pred_logits = jnp.mean(jax.vmap(
        jax.vmap(_linearized_apply, (None, 0)), (0, None)
    )(states, X_cb), axis=0)
    nll_lin = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
        ylin_pred_logits, y_cb
    ))
    
    # Linearized-misclassification
    ylin_preds = jnp.argmax(ylin_pred_logits, axis=-1)
    miscl_lin = jnp.mean(ylin_preds != y_cb)
    return nll_pi, miscl_pi, nll, miscl, nll_lin, miscl_lin


def initialize_model(key, features, x, model_type="mlp"):
    if isinstance(key, int):
        key = jr.key(key)
    key, subkey = jr.split(key)
    if model_type == "mlp":
        model = MLP(features=features)
    elif model_type == "cnn":
        model = CNN((1, 28, 28, 1), 10)
    params = model.init(key, x)
    flat_params, unflatten_fn = ravel_pytree(params)
    print(f"Number of parameters: {len(flat_params)}")
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


def tune_agents(key, agents, dlr_ranks, n_samples, n_iters, init_kwargs, 
                X_tune, Y_tune, tune_loss_fn, hparam_path, tune_niter):
    agent_queue = {}
    for agent in agents:
        if "dlr" in agent:
            for rank in dlr_ranks:
                curr_kwargs = {**init_kwargs}
                curr_kwargs['rank'] = rank
                agent_queue[f"{agent}-R{rank}"] = (agent, curr_kwargs)
        else:
            agent_queue[agent] = (agent, init_kwargs)
    tuned_agents = {}
    for agent, (agent_name, kwargs) in agent_queue.items():
        if any(lr_agent in agent for lr_agent in ('bbb', 'blr', 'bog')):
            hyperparams = ["learning_rate", "init_cov"]
        else:
            hyperparams = ["init_cov"]
        curr_kwargs = {**kwargs}
        if "-l-" in agent:
            curr_kwargs["emission_mean_function"] = \
                curr_kwargs["em_linpi_function"]
        curr_kwargs.pop("em_linpi_function")
        for n_sample in n_samples:
            for n_iter in n_iters:
                key, subkey = jr.split(key)
                curr_agent_str = f"{agent}-M{n_sample}-I{n_iter}"
                # Try to load hyperparameters
                curr_hparam_path = Path(hparam_path, f"{curr_agent_str}.json")
                if curr_hparam_path.exists():
                    with open(curr_hparam_path, "r") as f:
                        best_hparams = json.load(f)
                else:
                    print(f"Tuning {curr_agent_str}...")
                    curr_initializer = lambda **kkwargs: BONG_DICT[agent_name](
                        **kkwargs,
                        num_iter=n_iter,
                        num_samples=n_sample,
                    )
                    try:
                        best_hparams = tune_init_hyperparam(
                            key, curr_initializer, X_tune, Y_tune,
                            tune_loss_fn, hyperparams, minval=1e-5,
                            maxval=1.0, n_trials=tune_niter, **curr_kwargs
                        )
                    except:
                        best_hparams = {hparam: 1e-2 for hparam in hyperparams}
                    # Save hyperparameters
                    with open(curr_hparam_path, "w") as f:
                        json.dump(best_hparams, f)
                curr_agent = BONG_DICT[agent_name](
                    **best_hparams,
                    **curr_kwargs,
                    num_samples=n_sample,
                    num_iter=n_iter,
                )
                tuned_agents[agent] = curr_agent
    return tuned_agents, subkey


def evaluate_agents(key, agent_queue, X_tr, Y_tr, eval_cb_fn, eval_niter,
                    result_path):
    key, subkey = jr.split(key)
    result_dict = {}
    for agent_name, agent in agent_queue.items():
        curr_path = Path(result_path, f"{agent_name}.pkl")
        if curr_path.exists():
            with open(curr_path, "rb") as f:
                result_dict[agent_name] = pickle.load(f)
            continue
        print(f"Running {agent_name}...")
        nll_pis, miscl_pis, nlls, miscls, nll_lins, miscl_lins, runtimes = \
            [], [], [], [], [], [], []
        for _ in tqdm.trange(eval_niter):
            key1, key2, subkey = jr.split(subkey, 3)
            idx = jr.permutation(key1, jnp.arange(X_tr.shape[0]))
            X_tr, Y_tr = X_tr[idx], Y_tr[idx]
            _, (nll_pi, miscl_pi, nll, miscl, nll_lin, miscl_lin) = \
                jax.block_until_ready(
                    run_rebayes_algorithm(key2, agent, X_tr, Y_tr, 
                                          transform=eval_cb_fn)
                )
            # Run without callback to measure runtime
            t0 = time.perf_counter()
            _ = jax.block_until_ready(
                run_rebayes_algorithm(key2, agent, X_tr, Y_tr)
            )
            t1 = time.perf_counter()
            nll_pis.append(nll_pi)
            miscl_pis.append(miscl_pi)
            nlls.append(nll)
            miscls.append(miscl)
            nll_lins.append(nll_lin)
            miscl_lins.append(miscl_lin)
            runtimes.append(t1 - t0)
        nll_pis, miscl_pis, nlls, miscls, nll_lins, miscl_lins, runtimes = \
            _farray(nll_pis), _farray(miscl_pis), _farray(nlls), \
            _farray(miscls), _farray(nll_lins), _farray(miscl_lins), \
            jnp.array(runtimes)
        result_dict[agent_name] = \
            (runtimes, nll_pis, miscl_pis, nlls, miscls, nll_lins, miscl_lins)
        print(f"\tPlugin-NLL: {jnp.nanmean(nll_pis):.4f},"
              f" Plugin-Miscl: {jnp.nanmean(miscl_pis):.4f}")
        print(f"\tNLPD-NLL: {jnp.nanmean(nlls):.4f},"
              f" NLPD-Miscl: {jnp.nanmean(miscls):.4f}")
        print(f"\tLinearized-NLL: {jnp.nanmean(nll_lins):.4f},"
              f"Linearized-Miscl: {jnp.nanmean(miscl_lins):.4f}")
        print(f"\tTime: {t1 - t0:.2f}s")
        print("=====================================")
    return result_dict


def save_results(result_dict, result_path):
    for agent_name, agent_result in result_dict.items():
        with open(Path(result_path, f"{agent_name}.pkl"), "wb") as f:
            pickle.dump(agent_result, f)
    
    # Save runtime
    fig, ax = plt.subplots()
    for agent_name, (runtimes, *_) in result_dict.items():
        niter = len(runtimes)
        rt_mean = jnp.nanmean(runtimes, axis=0)
        rt_std = jnp.nanstd(runtimes, axis=0) / jnp.sqrt(niter)
        ax.bar(agent_name, rt_mean, yerr=rt_std, capsize=5)
    ax.set_ylabel("runtime (s)")
    plt.setp(ax.get_xticklabels(), rotation=60)
    fig.savefig(
        Path(result_path, "runtime.pdf"), bbox_inches='tight', dpi=300
    )
    
    # Save Plugin-NLL
    fig, ax = plt.subplots()
    for agent_name, (rts, nll_pis, *_) in result_dict.items():
        niter = len(nll_pis)
        rt_mean = jnp.nanmean(rts, axis=0)
        nll_pi_mean = jnp.nanmean(nll_pis, axis=0)
        nll_pi_std = jnp.nanstd(nll_pis, axis=0) / jnp.sqrt(niter)
        ax.plot(nll_pi_mean, label=f"{agent_name}[sec:{rt_mean:.1f}]")
        ax.fill_between(
            jnp.arange(nll_pi_mean.shape[0]), 
            nll_pi_mean - nll_pi_std, nll_pi_mean + nll_pi_std,
            alpha=0.3
        )
    ax.set_xlabel("num. training observations")
    ax.set_ylabel("nlpd (plugin)")
    ax.grid()
    ax.legend()
    fig.savefig(
        Path(result_path, "plugin_nlpd.pdf"), bbox_inches='tight', dpi=300
    )
    
    # Save Plugin-Misclassification
    fig, ax = plt.subplots()
    for agent_name, (rts, _, miscl_pis, *_) in result_dict.items():
        niter = len(miscl_pis)
        rt_mean = jnp.nanmean(rts, axis=0)
        miscl_pi_mean = jnp.nanmean(miscl_pis, axis=0)
        miscl_pi_std = jnp.nanstd(miscl_pis, axis=0) / jnp.sqrt(niter)
        ax.plot(miscl_pi_mean, label=f"{agent_name}[sec:{rt_mean:.1f}]")
        ax.fill_between(
            jnp.arange(miscl_pi_mean.shape[0]), 
            miscl_pi_mean - miscl_pi_std, miscl_pi_mean + miscl_pi_std,
            alpha=0.3
        )
    ax.set_xlabel("number of iteration")
    ax.set_ylabel("misclassification (plugin)")
    ax.grid()
    ax.legend()
    fig.savefig(
        Path(result_path, "plugin_miscl.pdf"), bbox_inches='tight', dpi=300
    )
    
    # Save NLPD-NLL
    fig, ax = plt.subplots()
    for agent_name, (rts, _, _, nlls, *_) in result_dict.items():
        niter = len(nlls)
        rt_mean = jnp.nanmean(rts, axis=0)
        nll_mean = jnp.nanmean(nlls, axis=0)
        nll_std = jnp.nanstd(nlls, axis=0) / jnp.sqrt(niter)
        ax.plot(nll_mean, label=f"{agent_name}[sec:{rt_mean:.1f}]")
        ax.fill_between(
            jnp.arange(nll_mean.shape[0]), 
            nll_mean - nll_std, nll_mean + nll_std,
            alpha=0.3
        )
    ax.set_xlabel("number of iteration")
    ax.set_ylabel("nlpd (MC)")
    ax.grid()
    ax.legend()
    fig.savefig(
        Path(result_path, "mc_nlpd.pdf"), bbox_inches='tight', dpi=300
    )
    
    # Save NLPD-Misclassifciation
    fig, ax = plt.subplots()
    for agent_name, (rts, _, _, _, miscls, *_) in result_dict.items():
        niter = len(miscls)
        rt_mean = jnp.nanmean(rts, axis=0)
        miscl_mean = jnp.nanmean(miscls, axis=0)
        miscl_std = jnp.nanstd(miscls, axis=0) / jnp.sqrt(niter)
        ax.plot(miscl_mean, label=f"{agent_name}[sec:{rt_mean:.1f}]")
        ax.fill_between(
            jnp.arange(miscl_mean.shape[0]), 
            miscl_mean - miscl_std, miscl_mean + miscl_std,
            alpha=0.3
        )
    ax.set_xlabel("number of iteration")
    ax.set_ylabel("misclassification (MC)")
    ax.grid()
    ax.legend()
    fig.savefig(
        Path(result_path, "mc_miscl.pdf"), bbox_inches='tight', dpi=300
    )
    
    # Save Linearized-NLL
    fig, ax = plt.subplots()
    for agent_name, (rts, _, _, _, _, nll_lins, *_) in result_dict.items():
        niter = len(nll_lins)
        rt_mean = jnp.nanmean(rts, axis=0)
        nll_lin_mean = jnp.nanmean(nll_lins, axis=0)
        nll_lin_std = jnp.nanstd(nll_lins, axis=0) / jnp.sqrt(niter)
        ax.plot(nll_lin_mean, label=f"{agent_name}[sec:{rt_mean:.1f}]")
        ax.fill_between(
            jnp.arange(nll_lin_mean.shape[0]), 
            nll_lin_mean - nll_lin_std, nll_lin_mean + nll_lin_std,
            alpha=0.3
        )
    ax.set_xlabel("number of iteration")
    ax.set_ylabel("nlpd (lin-MC)")
    ax.grid()
    ax.legend()
    fig.savefig(
        Path(result_path, "linmc_nlpd.pdf"), bbox_inches='tight', dpi=300
    )
    
    # Save Linearized-Accuracy
    fig, ax = plt.subplots()
    for agent_name, (rts, _, _, _, _, _, miscl_lins) in result_dict.items():
        niter = len(miscl_lins)
        rt_mean = jnp.nanmean(rts, axis=0)
        miscl_lin_mean = jnp.nanmean(miscl_lins, axis=0)
        miscl_lin_std = jnp.nanstd(miscl_lins, axis=0) / jnp.sqrt(niter)
        ax.plot(miscl_lin_mean, label=f"{agent_name}[sec:{rt_mean:.1f}]")
        ax.fill_between(
            jnp.arange(miscl_lin_mean.shape[0]), 
            miscl_lin_mean - miscl_lin_std, miscl_lin_mean + miscl_lin_std,
            alpha=0.3
        )
    ax.set_xlabel("number of iteration")
    ax.set_ylabel("misclassification (lin-MC)")
    ax.grid()
    ax.legend()
    fig.savefig(
        Path(result_path, "linmc_miscl.pdf"), bbox_inches='tight', dpi=300
    )
    
    plt.close('all')
    

def main(args):
    # Generate dataset
    X_tr, Y_tr, X_val, Y_val, X_te, Y_te = load_mnist_dataset(
        args.n_train, args.n_test
    )
    
    # Set up MLP model
    model_init_kwargs, key = \
        initialize_model(args.seed, args.features, X_tr[0], args.model)
    em_function = model_init_kwargs["emission_mean_function"]
    
    # Tune hyperparameters
    hparam_path = Path(hparam_mnist_path, args.model)
    hparam_path.mkdir(parents=True, exist_ok=True)
    tune_loss_fn = partial(loss_fn, em_function=em_function,
                           X_val=X_val, y_val=Y_val)
    agent_queue, key = tune_agents(
        key, args.agents, args.dlr_ranks, args.num_samples, args.num_iters, 
        model_init_kwargs, X_tr, Y_tr, tune_loss_fn, 
        hparam_path, args.tune_niter
    )
    
    # Evaluate agents
    result_path = Path(mnist_path, args.model)
    result_path.mkdir(parents=True, exist_ok=True)
    eval_cb_fn = partial(callback_fn, em_function=em_function,
                         X_cb=X_te, y_cb=Y_te)
    result_dict = evaluate_agents(key, agent_queue, X_tr, Y_tr, eval_cb_fn,
                                  args.eval_niter, result_path)
    
    # Save results
    save_results(result_dict, result_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Data parameters
    parser.add_argument("--n_train", type=int, default=2_000)
    parser.add_argument("--n_test", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    
    # Model parameters
    parser.add_argument("--agents", type=str, nargs="+",
                        default=["dg-bong"], choices=BONG_DICT.keys())
    parser.add_argument("--dlr_ranks", type=int, nargs="+", default=[10,])
    parser.add_argument("--num_samples", type=int, nargs="+", 
                        default=[100])
    parser.add_argument("--num_iters", type=int, nargs="+", 
                        default=[10])
    parser.add_argument("--init_var", type=float, default=4.0)
    parser.add_argument("--laplace_gtol", type=float, default=1e-3)
    parser.add_argument("--model", type=str, default="mlp",
                        choices=["mlp", "cnn"])
    parser.add_argument("--features", type=int, nargs="+", default=[50, 50, 10])
    
    # Eval parameters
    parser.add_argument("--tune_niter", type=int, default=30)
    parser.add_argument("--eval_niter", type=int, default=10)
    
    args = parser.parse_args()
    main(args)