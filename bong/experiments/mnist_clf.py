import argparse
from functools import partial
import json
from pathlib import Path
import time

import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax

from bong.agents import DIAG_BONG_DICT, DLR_BONG_DICT
from bong.settings import hparam_mnist_path, mnist_path
import bong.src.dataloaders as dataloaders
from bong.util import MLP, run_rebayes_algorithm, tune_init_hyperparam


BONG_DICT = {**DIAG_BONG_DICT, **DLR_BONG_DICT}


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
    # Plugin-LL
    ypi_pred_logits = jax.vmap(em_function, (None, 0))(state.mean, X_cb)
    ll_pi = jnp.mean(-optax.softmax_cross_entropy_with_integer_labels(
        ypi_pred_logits, y_cb
    ))
    
    # Plugin-accuracy
    ypi_preds = jnp.argmax(ypi_pred_logits, axis=-1)
    acc_pi = jnp.mean(ypi_preds == y_cb)
    
    # NLPD-LL
    states = alg.sample(key, state, num_samples)
    y_pred_logits = jnp.mean(jax.vmap(
        jax.vmap(em_function, (None, 0)), (0, None)
    )(states, X_cb), axis=0)
    ll = jnp.mean(-optax.softmax_cross_entropy_with_integer_labels(
        y_pred_logits, y_cb
    ))
    
    # NLPD-accuracy
    y_preds = jnp.argmax(y_pred_logits, axis=-1)
    acc = jnp.mean(y_preds == y_cb)
    return ll_pi, acc_pi, ll, acc


def initialize_mlp_model(key, features, x):
    if isinstance(key, int):
        key = jr.key(key)
    key, subkey = jr.split(key)
    model = MLP(features=features)
    params = model.init(key, x)
    flat_params, unflatten_fn = ravel_pytree(params)
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
        "emission_cov_function": ec_function,
    }
    return init_kwargs, subkey


def tune_agents(key, agents, n_samples, n_iters, init_kwargs, 
                X_tune, Y_tune, tune_loss_fn):
    agent_queue = {}
    for agent in agents:
        if any(lr_agent in agent for lr_agent in ('bbb', 'blr', 'bog')):
            hyperparams = ["learning_rate", "init_cov"]
        else:
            hyperparams = ["init_cov"]
        for n_sample in n_samples:
            for n_iter in n_iters:
                key, subkey = jr.split(key)
                curr_agent_str = f"{agent}-M{n_sample}-I{n_iter}"
                # Try to load hyperparameters
                hparam_path = Path(hparam_mnist_path, f"{curr_agent_str}.json")
                if hparam_path.exists():
                    with open(hparam_path, "r") as f:
                        best_hparams = json.load(f)
                else:
                    curr_initializer = lambda **kwargs: BONG_DICT[agent](
                        **kwargs,
                        num_iter=n_iter,
                        num_samples=n_sample,
                    )
                    try:
                        best_hparams = tune_init_hyperparam(
                            key, curr_initializer, X_tune, Y_tune,
                            tune_loss_fn, hyperparams, minval=1e-5,
                            maxval=1.0, **init_kwargs
                        )
                    except:
                        best_hparams = {hparam: 1e-2 for hparam in hyperparams}
                    # Save hyperparameters
                    with open(hparam_path, "w") as f:
                        json.dump(best_hparams, f)
                curr_agent = BONG_DICT[agent](
                    **best_hparams,
                    **init_kwargs,
                    num_samples=n_sample,
                    num_iter=n_iter,
                )
                agent_queue[f"{agent}-M{n_sample}-I{n_iter}-tuned"] = curr_agent
    return agent_queue, subkey


def evaluate_agents(key, agent_queue, X_tr, Y_tr, eval_cb_fn):
    result_dict = {}
    for agent_name, agent in agent_queue.items():
        print(f"Running {agent_name}...")
        key, subkey = jr.split(key)
        t0 = time.perf_counter()
        _, (ll_pi, acc_pi, ll, acc) = jax.block_until_ready(
            run_rebayes_algorithm(key, agent, X_tr, Y_tr, 
                                  transform=eval_cb_fn)
        )
        t1 = time.perf_counter()
        result_dict[agent_name] = (t1 - t0, ll_pi, acc_pi, ll, acc)
        print(f"\tPlugin-LL: {ll_pi[-1]:.4f}, Plugin-Acc: {acc_pi[-1]:.4f}")
        print(f"\tNLPD-LL: {ll[-1]:.4f}, NLPD-Acc: {acc[-1]:.4f}")
        print(f"\tTime: {t1 - t0:.2f}s")
        print("=====================================")
    return result_dict


def save_results(result_dict):
    # Save runtime
    fig, ax = plt.subplots()
    for agent_name, (runtime, _, _, _, _) in result_dict.items():
        ax.bar(agent_name, runtime)
    ax.set_ylabel("runtime (s)")
    plt.setp(ax.get_xticklabels(), rotation=30)
    fig.savefig(
        Path(mnist_path, "runtime.pdf"), bbox_inches='tight', dpi=300
    )
    
    # Save Plugin-LL
    fig, ax = plt.subplots()
    for agent_name, (_, ll_pi, _, _, _) in result_dict.items():
        if jnp.any(jnp.isnan(ll_pi)):
            continue
        ax.plot(ll_pi, label=agent_name)
    ax.set_xlabel("number of iteration")
    ax.set_ylabel("LL (plugin)")
    ax.grid()
    ax.legend()
    fig.savefig(
        Path(mnist_path, "plugin_ll.pdf"), bbox_inches='tight', dpi=300
    )
    
    # Save Plugin-Accuracy
    fig, ax = plt.subplots()
    for agent_name, (_, _, acc_pi, _, _) in result_dict.items():
        if jnp.any(jnp.isnan(acc_pi)):
            continue
        ax.plot(acc_pi, label=agent_name)
    ax.set_xlabel("number of iteration")
    ax.set_ylabel("Accuracy (plugin)")
    ax.grid()
    ax.legend()
    fig.savefig(
        Path(mnist_path, "plugin_acc.pdf"), bbox_inches='tight', dpi=300
    )
    
    # Save NLPD-LL
    fig, ax = plt.subplots()
    for agent_name, (_, _, _, ll, _) in result_dict.items():
        if jnp.any(jnp.isnan(ll)):
            continue
        ax.plot(ll, label=agent_name)
    ax.set_xlabel("number of iteration")
    ax.set_ylabel("LL (NLPD)")
    ax.grid()
    ax.legend()
    fig.savefig(
        Path(mnist_path, "nlpd_ll.pdf"), bbox_inches='tight', dpi=300
    )
    
    # Save NLPD-Accuracy
    fig, ax = plt.subplots()
    for agent_name, (_, _, _, _, acc) in result_dict.items():
        if jnp.any(jnp.isnan(acc)):
            continue
        ax.plot(acc, label=agent_name)
    ax.set_xlabel("number of iteration")
    ax.set_ylabel("Accuracy (NLPD)")
    ax.grid()
    ax.legend()
    fig.savefig(
        Path(mnist_path, "nlpd_acc.pdf"), bbox_inches='tight', dpi=300
    )
    
    plt.close('all')
    

def main(args):
    # Generate dataset
    X_tr, Y_tr, X_val, Y_val, X_te, Y_te = load_mnist_dataset(
        args.n_train, args.n_test
    )
    
    # Set up MLP model
    model_init_kwargs, key = \
        initialize_mlp_model(args.seed, args.features, X_tr[0])
    em_function = model_init_kwargs["emission_mean_function"]
    
    # Tune hyperparameters
    hparam_mnist_path.mkdir(parents=True, exist_ok=True)
    tune_loss_fn = partial(loss_fn, em_function=em_function,
                           X_val=X_val, y_val=Y_val)
    agent_queue, key = tune_agents(
        key, args.agents, args.num_samples, args.num_iters, 
        model_init_kwargs, X_tr, Y_tr, tune_loss_fn
    )
    
    # Evaluate agents
    eval_cb_fn = partial(callback_fn, em_function=em_function,
                         X_cb=X_te, y_cb=Y_te)
    result_dict = evaluate_agents(key, agent_queue, X_tr, Y_tr, eval_cb_fn)
    
    # Save results
    mnist_path.mkdir(parents=True, exist_ok=True)
    save_results(result_dict)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Data parameters
    parser.add_argument("--n_train", type=int, default=2_000)
    parser.add_argument("--n_test", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    
    # Model parameters
    parser.add_argument("--agents", type=str, nargs="+",
                        default=["dg-bong"], choices=BONG_DICT.keys())
    parser.add_argument("--num_samples", type=int, nargs="+", 
                        default=[100])
    parser.add_argument("--num_iters", type=int, nargs="+", 
                        default=[10])
    parser.add_argument("--init_var", type=float, default=4.0)
    parser.add_argument("--laplace_gtol", type=float, default=1e-3)
    parser.add_argument("--features", type=int, nargs="+", default=[50, 50, 10])
    
    args = parser.parse_args()
    main(args)