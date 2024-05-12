

import argparse
import os
import itertools
import pandas as pd
from pathlib import Path
import os

from bong.agents import AGENT_DICT, AGENT_NAMES, make_agent_name
from datasets import DATASET_NAMES, make_dataset_name
from bong.util import safestr
from plot_utils import plot_results_from_files

def main(args):
    if args.dir == "":
        data_name = make_dataset_name(args)
        agent_name = make_agent_name(args)
        path = Path(args.rootdir, data_name, agent_name)
    else:
        path = Path(args.dir)
    results_dir = str(path)
    print(f'Reading {results_dir}')
    for metric in ['kl', 'nll', 'nlpd']:
        plot_results_from_files(results_dir,  metric, save_fig=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir", type=str, default="/teamspace/studios/this_studio/jobs") 
    parser.add_argument("--dir", type=str, default="")
    parser.add_argument("--parallel", type=bool, default=False)

    # Data parameters
    parser.add_argument("--dataset", type=str, default="linreg", choices=DATASET_NAMES)
    parser.add_argument("--data_dim", type=int,  default=10)
    parser.add_argument("--data_key", type=int,  default=0)
    #parser.add_argument("--data_dim_list", type=int, nargs="+", default=[10])
    #parser.add_argument("--data_key_list", type=int, nargs="+", default=[0])
    
    # Agent parameters
    parser.add_argument("--agent_list", type=str, nargs="+", default=["bong_fc"], choices=AGENT_NAMES)
    parser.add_argument("--lr_list", type=float, nargs="+", default=[0.01])
    parser.add_argument("--niter_list", type=int, nargs="+", default=[10])
    parser.add_argument("--nsample_list", type=int, nargs="+", default=[10])
    parser.add_argument("--ef_list", type=int, nargs="+", default=[1])
    parser.add_argument("--rank_list", type=int, nargs="+", default=[10])
    parser.add_argument("--model_neurons_list", type=str, nargs="+",  default=["1"]) # 10-10-1


    args = parser.parse_args()
    
    main(args)