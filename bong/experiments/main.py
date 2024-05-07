import argparse
from functools import partial
from pathlib import Path
import time
import json
import pandas as pd
import jax.numpy as jnp
import numpy as np
import jax.random as jr

def make_results(args):
    # dummy work
    N = 1000
    key = jr.PRNGKey(0)
    M = jr.normal(key, (N,N))
    M = M * M

    # Save dummy results in dataframe
    T = 10
    steps = jnp.arange(0, T)
    kl = steps * args.learning_rate 
    nll = steps * steps * args.learning_rate 
    nlpd = nll
    df  = pd.DataFrame({'time': 0, 
                        'step': steps,
                        'kl': kl,
                        'nll': nll,
                        'nlpd': nlpd
    })
    return df

def extract_args_dict(args, parser):
    args_dict = {action.dest:  getattr(args, action.dest, None) for action in parser._actions}
    args_dict.pop('help')
    return args_dict

def main(args, args_dict):
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
    parser.add_argument("--dataset", type=str, default="logreg")
    parser.add_argument("--key", type=int, default=0)
    
    # Model parameters
    parser.add_argument("--agent", type=str, default="fg-bong")
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--num_iter", type=int, default=10) 

    # results
    parser.add_argument("--dir", type=str, default="", help="directory to store results") 
    parser.add_argument("--filename", type=str, default="", help="filename prefix")
    
    args = parser.parse_args()
    if False: #args.filename == "":
        lr_str = f"{round(args.learning_rate,4)}".replace('.', '_')
        args.filename = f"{args.dataset}_{args.agent}_LR{lr_str}_I{args.num_iter}_"  

    args_dict = extract_args_dict(args, parser)
    main(args, args_dict)

'''
python  experiments/main.py  --agent fg-bong --learning_rate 0.01 --num_iter 10 

'''