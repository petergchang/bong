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

from bong.util import run_rebayes_algorithm, get_gpu_name, unmake_neuron_str
from bong.agents import AGENT_DICT, AGENT_NAMES, parse_agent_full_name
from datasets import make_dataset
from models import make_model

def run_agent(key, agent, data, model):
    print(f"Running {agent.name} + {model['name']} on {data['name']}")
    print("Using GPU of type: ", get_gpu_name())
    t0 = time.perf_counter()
    _, output = jax.block_until_ready(
        run_rebayes_algorithm(key, agent, data['X_tr'], data['Y_tr'],
        transform=model['callback'])
    )
    results, summary = model['process_callback'](output)
    t1 = time.perf_counter()
    elapsed = t1-t0
    print(f"Time {elapsed:.2f}s")
    print(summary)
    #results['time'] = elapsed
    #results['agent_name'] = agent.name
    #results['dataset_name'] = data['name']
    return results, elapsed


def add_column_of_ones(A):
    ones_column = np.ones((A.shape[0], 1))
    A_with_ones = np.hstack((A, ones_column))
    return A_with_ones

def add_ones_to_covariates(data):
    data['X_tr'] = add_column_of_ones(data['X_tr'])
    data['X_val'] = add_column_of_ones(data['X_val'])
    data['X_te'] = add_column_of_ones(data['X_te'])
    return data

def make_results(args):
    data = make_dataset(args)
    if args.add_ones:
        data = add_ones_to_covariates(data)
        args.data_dim = args.data_dim + 1

    model = make_model(args, data)

    constructor = AGENT_DICT[args.agent]['constructor']
    agent = constructor(
                        **model['model_kwargs'],
                        agent_key = args.agent_key,
                        learning_rate = args.lr,
                        num_iter = args.niter,
                        num_samples = args.nsample,
                        linplugin = args.linplugin,
                        empirical_fisher = args.ef,
                        rank = args.rank
                    )
    key = jr.PRNGKey(args.agent_key)
    results, elapsed = run_agent(key, agent, data, model)
    df = pd.DataFrame(results)
    attributes = parse_agent_full_name(agent.name)
    meta = { # non time-series data
        'data_name': data['name'],
        'model_name': model['name'],
        'agent_name': agent.name,
        'algo': attributes['algo'],
        'param': attributes['param'],
        'elapsed': elapsed
        }
    return df, meta


def main(args, args_dict):
    if args.dgp_neurons_str == "":
        assert len(args.dgp_neurons) >= 1
    else:
        args.dgp_neurons = unmake_neuron_str(args.dgp_neurons_str)
        args_dict['dgp_neurons'] = args.dgp_neurons

    if args.model_neurons_str == "":
        assert len(args.model_neurons) >= 1
    else:
        args.model_neurons = unmake_neuron_str(args.model_neurons_str)
        args_dict['model_neurons'] = args.model_neurons

    results_path = Path(args.dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    print("Saving single job results to", args.dir)
    # Make sure we can write to the output directory before doing real work\
    fname = Path(results_path, f"dummy.txt")
    with open(fname, 'w') as file:
        # Write an empty string to the file
        file.write('')

    df, meta = make_results(args)

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
    parser.add_argument("--data_key", type=int, default=0)
    parser.add_argument("--dgp_type", type=str, default="lin") # or mlp
    parser.add_argument("--dgp_neurons", type=int, nargs="+", default=[20, 20, 1]) 
    parser.add_argument("--dgp_neurons_str", type=str, default="") # 20_20_1 
    parser.add_argument("--emission_noise", type=float, default=1.0)
    parser.add_argument("--ntrain", type=int, default=500)
    parser.add_argument("--nval", type=int, default=500)
    parser.add_argument("--ntest", type=int, default=1000)
    parser.add_argument("--add_ones", type=int, default=0)

    
    # Model parameters
    parser.add_argument("--agent", type=str, default="bong_fc", choices=AGENT_NAMES)
    parser.add_argument("--agent_key", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--niter", type=int, default=10) 
    parser.add_argument("--nsample", type=int, default=100) 
    parser.add_argument("--ef", type=int, default=1)
    parser.add_argument("--linplugin", type=int, default=0)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--model_type", type=str, default="lin") # or mlp
    parser.add_argument("--model_neurons", type=int, nargs="+", default=[10, 10, 1])
    parser.add_argument("--model_neurons_str", type=str, default="")
    parser.add_argument("--use_bias", type=int, default=1) 
    parser.add_argument("--init_var", type=float, default=1.0)
    parser.add_argument("--algo_key", type=int, default=0)

    # results
    parser.add_argument("--dir", type=str, default="", help="directory to store results") 
    parser.add_argument("--debug", type=bool, default=False)
    
    args = parser.parse_args()
     # Convert parser flags to dictionary
    args_dict = {action.dest:  getattr(args, action.dest, None) for action in parser._actions}
    args_dict.pop('help')
    main(args, args_dict)

