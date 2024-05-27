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
import jax.numpy as jnp

from bong.util import run_rebayes_algorithm, get_gpu_name
#from bong.agents import AGENT_DICT, AGENT_NAMES, parse_agent_full_name, make_agent_name_from_parts
from bong.agents import make_agent_constructor
from datasets import make_dataset
from models import make_model

def run_agent(key, agent, data, model):
    sh = jnp.array(data['X_tr'].shape)
    N = sh[0]
    D = jnp.prod(sh[1:])
    print(f"Running {agent.name} + {model['name']} on {data['name']} of size N={N}, D={D}")
    print("Using GPU of type: ", get_gpu_name())
    t0 = time.perf_counter()
    _, output = jax.block_until_ready(
        run_rebayes_algorithm(key, agent, data['X_tr'], data['Y_tr'],
        transform=model['callback'])
    )
    results, summary = model['process_callback'](output)
    t1 = time.perf_counter()
    elapsed = t1-t0
    print(f"Scan time {elapsed:.2f}s")
    print(summary)
    return results, elapsed, summary


def make_results(key, args):
    key, subkey = jr.split(key)
    key, data = make_dataset(subkey, args)
    key, subkey = jr.split(key)
    key, model = make_model(subkey, args, data)

    #name = f'{args.algo}_{args.param}' # eg bong-fc_mom, must match keys of AGENT_DICT
    #constructor = AGENT_DICT[name]['constructor']
    constructor = make_agent_constructor(args.algo, args.param)
    key, subkey = jr.split(key)
    agent = constructor(
                        **model['model_kwargs'],
                        agent_key = subkey,
                        learning_rate = args.lr,
                        num_iter = args.niter,
                        num_samples = args.nsample,
                        linplugin = args.lin,
                        empirical_fisher = args.ef,
                        rank = args.rank
                    )
    key, subkey = jr.split(key)
    results, elapsed, summary = run_agent(subkey, agent, data, model)
    df = pd.DataFrame(results)
    meta = { # non time-series data
        'data_name': data['name'],
        'data_dim': args.data_dim,
        'ntrain': args.ntrain,
        'ntest': args.ntest,
        'model_name': model['name'],
        'model_nparams': model['nparams'],
        'agent_name': agent.name,
        'agent_full_name': agent.full_name,
        'elapsed': elapsed,
        'summary': summary,
        'metrics': df.columns
        }
    return df, meta, key


def main(args, args_dict):
    key = jr.PRNGKey(args.seed)
    results_path = Path(args.dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    print("Saving single job results to", results_path)
    # Make sure we can write to the output directory before doing real work\
    fname = Path(results_path, f"dummy.txt")
    with open(fname, 'w') as file:
        # Write an empty string to the file
        file.write('')

    df, meta, key = make_results(key, args)

    fname = Path(results_path, f"args.json")
    args_dict = args_dict | meta
    with open(fname, 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)

    fname = Path(results_path, f"results.csv")
    df.to_csv(fname, index=False) #, na_rep="NAN", mode="w")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data parameters
    parser.add_argument("--dataset", type=str, default="reg") 
    parser.add_argument("--data_dim", type=int, default=10)
    parser.add_argument("--dgp_type", type=str, default="lin") # or mlp
    parser.add_argument("--dgp_str", type=str, default="") # 20_20_1 
    parser.add_argument("--emission_noise", type=float, default=-1)
    parser.add_argument("--ntrain", type=int, default=1000)
    parser.add_argument("--nval", type=int, default=1000)
    parser.add_argument("--ntest", type=int, default=1000)
    parser.add_argument("--add_ones", type=int, default=0)

    
    # Model parameters
    #parser.add_argument("--agent", type=str, default="bong_fc", choices=AGENT_NAMES)
    parser.add_argument("--algo", type=str, default="bong")
    parser.add_argument("--param", type=str, default="fc")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--niter", type=int, default=10) 
    parser.add_argument("--nsample", type=int, default=100) 
    parser.add_argument("--ef", type=int, default=1)
    parser.add_argument("--lin", type=int, default=0)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--model_type", type=str, default="lin") # or mlp
    parser.add_argument("--model_str", type=str, default="")
    parser.add_argument("--use_bias", type=int, default=1) 
    parser.add_argument("--use_bias_layer1", type=int, default=1) 
    parser.add_argument("--init_var", type=float, default=1)

    # results
    parser.add_argument("--dir", type=str, default="", help="directory to store results") 
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    
    args = parser.parse_args()
     # Convert parser flags to dictionary
    args_dict = {action.dest:  getattr(args, action.dest, None) for action in parser._actions}
    args_dict.pop('help')
    main(args, args_dict)

