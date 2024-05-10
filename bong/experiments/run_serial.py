

import argparse
import os
import itertools
import pandas as pd
from pathlib import Path

from bong.agents import AGENT_DICT, AGENT_NAMES
from job_utils import make_cmd_dict_for_flag_crossproduct, make_results_dirname


def main(args):
    cmd_dict = make_cmd_dict_for_flag_crossproduct(args)

    for jobname, cmd in cmd_dict.items():   
        output_dir = make_results_dirname(jobname, args.parallel)         
        cmd = cmd + f' --dir {output_dir}'
        print('running', cmd)
        os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="", help="directory in which to store jobs.csv") 
    parser.add_argument("--parallel", type=bool, default=False)

    # Data parameters
    parser.add_argument("--dataset", type=str, default="linreg")
    parser.add_argument("--key", type=int, default=0)
    parser.add_argument("--ntrain", type=int, default=500)
    parser.add_argument("--nval", type=int, default=500)
    parser.add_argument("--ntest", type=int, default=500)
    parser.add_argument("--data_dim", type=int, default=10)
    parser.add_argument("--emission_noise", type=float, default=1.0)
    

    parser.add_argument("--agents", type=str, nargs="+",
                        default=["bong-fc", "blr-fc"], choices=AGENT_NAMES)
    parser.add_argument("--lrs", type=float, nargs="+", 
                    default=[0.01, 0.05])
    parser.add_argument("--niters", type=int, nargs="+", 
                    default=[10])
    parser.add_argument("--nsamples", type=int, nargs="+", 
                    default=[10])

    args = parser.parse_args()
    
    main(args)

'''
python run_serial.py  --lrs 0.001 0.01 --agents blr-fc --dir ~/jobs/foo
'''