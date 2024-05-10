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
from linreg_data import make_linreg
from job_utils import  run_agent



def make_results(args):
    if args.dataset == "linreg":
        data, init_kwargs, callback, tune_fn = make_linreg(args)
    else:
        raise Exception(f'unrecognized dataset {args.dataset}')

    constructor = AGENT_DICT[args.agent]['constructor']
    agent = constructor(
                        **init_kwargs,
                        learning_rate = args.lr,
                        num_samples = args.nsample,
                        num_iter = args.niter,
                        linplugin = args.linplugin,
                        empirical_fisher = args.ef
                    )
    key = jr.PRNGKey(args.agent_key)
    results = run_agent(key, agent, data, callback)
    df = pd.DataFrame(results)
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
    parser.add_argument("--data_key", type=int, default=0)
    parser.add_argument("--ntrain", type=int, default=500)
    parser.add_argument("--nval", type=int, default=500)
    parser.add_argument("--ntest", type=int, default=500)
    parser.add_argument("--data_dim", type=int, default=10)
    parser.add_argument("--emission_noise", type=float, default=1.0)
    
    # Model parameters
    parser.add_argument("--agent", type=str, default="bong-fc", choices=AGENT_NAMES)
    parser.add_argument("--agent_key", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--niter", type=int, default=10) 
    parser.add_argument("--nsample", type=int, default=10) 
    parser.add_argument("--ef", type=int, default=1)
    parser.add_argument("--linplugin", type=int, default=0)

    # results
    parser.add_argument("--dir", type=str, default="", help="directory to store results") 
    parser.add_argument("--filename", type=str, default="", help="filename prefix")
    
    args = parser.parse_args()
    args_dict = extract_args_dict(args, parser)
    main(args, args_dict)

'''
python run_one_job.py   --agent bong-fc --dataset linreg --ntrain 10 --ef 0 --dir ~/jobs/linreg-bong-new
'''
