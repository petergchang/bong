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

from bong.util import run_rebayes_algorithm, get_gpu_name
from bong.agents import AGENT_DICT, AGENT_NAMES
from datasets import make_dataset
from models import make_model

def run_agent(key, agent, data, callback):
    print(f"Running {agent.name} on {data['name']}")
    print("Using GPU of type: ", get_gpu_name())
    t0 = time.perf_counter()
    _, (kldiv, nll, nlpd) = jax.block_until_ready(
        run_rebayes_algorithm(key, agent, data['X_tr'], data['Y_tr'], transform=callback)
    )
    t1 = time.perf_counter()
    print(f"KL-Div: {kldiv[-1]:.4f}, NLL: {nll[-1]:.4f},  NLPD: {nlpd[-1]:.4f}, Time: {t1 - t0:.2f}s")
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
    data = make_dataset(args)
    init_kwargs, callback = make_model(args, data)

    constructor = AGENT_DICT[args.agent]['constructor']
    agent = constructor(
                        **init_kwargs,
                        agent_key = args.agent_key,
                        learning_rate = args.lr,
                        num_iter = args.niter,
                        num_samples = args.nsample,
                        linplugin = args.linplugin,
                        empirical_fisher = args.ef,
                        rank = args.rank
                    )
    key = jr.PRNGKey(args.agent_key)
    results = run_agent(key, agent, data, callback)
    df = pd.DataFrame(results)
    return df


def main(args, args_dict):
    #print(args)
    results_path = Path(args.dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Make sure we can write to the output directory before doing real work
    fname = Path(results_path, f"{args.filename}args.json")
    print("Saving to", fname)
    with open(fname, 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)

    df = make_results(args)

    fname = Path(results_path, f"{args.filename}results.csv")
    #print("Saving to", fname)
    df.to_csv(fname, index=False) #, na_rep="NAN", mode="w")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data parameters
    parser.add_argument("--dataset", type=str, default="reg") 
    parser.add_argument("--data_dim", type=int, default=10)
    parser.add_argument("--dgp", type=str, default="lin_1") # or mlp_20_20_1
    parser.add_argument("--emission_noise", type=float, default=1.0)
    parser.add_argument("--data_key", type=int, default=0)
    parser.add_argument("--ntrain", type=int, default=500)
    parser.add_argument("--nval", type=int, default=500)
    parser.add_argument("--ntest", type=int, default=500)

    
    # Model parameters
    parser.add_argument("--agent", type=str, default="bong_fc", choices=AGENT_NAMES)
    parser.add_argument("--agent_key", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--niter", type=int, default=10) 
    parser.add_argument("--nsample", type=int, default=10) 
    parser.add_argument("--ef", type=int, default=1)
    parser.add_argument("--linplugin", type=int, default=0)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--model", type=str, default="lin_1") # or mlp_10_10_1

    # results
    parser.add_argument("--dir", type=str, default="", help="directory to store results") 
    parser.add_argument("--filename", type=str, default="", help="filename prefix")
    parser.add_argument("--debug", type=bool, default=False)
    
    args = parser.parse_args()
     # Convert parser flags to dictionary
    args_dict = {action.dest:  getattr(args, action.dest, None) for action in parser._actions}
    args_dict.pop('help')
    main(args, args_dict)

'''
python run_job.py   --agent bong-fc --dataset linreg --data_dim 100 --ef 0 --dir ~/jobs/linreg-bong
'''
