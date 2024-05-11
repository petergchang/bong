import argparse
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

#from bong.settings import linreg_path
from bong.src import bbb, blr, bog, bong, experiment_utils
from bong.util import run_rebayes_algorithm, tune_init_hyperparam, gaussian_kl_div
from plot_utils import plot_results
from job_utils import convert_result_dict_to_pandas, split_filename_column,  run_agents

from bong.agents import AGENT_TYPES, LR_AGENT_TYPES, BONG_DICT
from linreg_data import make_linreg



def make_agent_queue(subkey, args, init_kwargs, tune_kl_loss_fn, X_tune, Y_tune):
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
                            key, curr_initializer, X_tune, Y_tune,
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


def make_results_dict(args):
    data, init_kwargs, callback, tune_obj_fn = make_linreg(args)
    X_tune, Y_tune = data['X_tr'], data['Y_tr']
    key = jr.PRNGKey(args.agent_key)
    agent_queue, subkey = make_agent_queue(key, args, init_kwargs, tune_obj_fn, X_tune, Y_tune)
    key = jr.PRNGKey(args.agent_key)
    result_dict = run_agents(key, agent_queue, data, callback)
    return result_dict
    #return result_dict, data, init_kwargs, callback

def main(args):
    result_dict = make_results_dict(args)
    
    curr_path = Path(args.dir)
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
    


   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Data parameters
    parser.add_argument("--dataset", type=str, default="linreg")
    parser.add_argument("--data_key", type=int, default=0)
    parser.add_argument("--ntrain", type=int, default=500)
    parser.add_argument("--nval", type=int, default=500)
    parser.add_argument("--ntest", type=int, default=500)
    parser.add_argument("--data_dim", type=int, default=10)
    parser.add_argument("--emission_noise", type=float, default=1.0)

    
    # Model parameters
    parser.add_argument("--agents", type=str, nargs="+",
                        default=["fg-bong"], choices=AGENT_TYPES)
    parser.add_argument("--agent_key", type=int, default=0)
    parser.add_argument("--num_samples", type=int, nargs="+", 
                        default=[10,])
    
    parser.add_argument("--learning_rate", type=float, nargs="+", 
                    default=[0.005, 0.01, 0.05])
    parser.add_argument("--num_iter", type=int, nargs="+", 
                    default=[10])
    parser.add_argument("--tune_learning_rate", type=bool, default=False)


    parser.add_argument("--filename", type=str, default="")
    parser.add_argument("--dir", type=str, default="")

    args = parser.parse_args()
    main(args)

'''
python  linreg_run.py  --agents fg-bong   --ntrain 10 --filename linreg-bong  --dir ~/jobs
'''