import argparse
import os
import itertools
import pandas as pd
from pathlib import Path
import os
import datetime
import json

from bong.util import  move_df_col
from job_utils import make_unix_cmd_given_flags, make_df_crossproduct

def main(args):
    path = Path(args.dir)
    results_dir = str(path)
    print(f'Saving jobs to {results_dir}')
    path.mkdir(parents=True, exist_ok=True)

    df = make_df_crossproduct(
        args.algo_list, args.param_list, args.lin_list,
        args.lr_list, args.niter_list, args.nsample_list,
        args.ef_list, args.rank_list, args.model_str_list)

    N = len(df)
    jobnames = [f'{args.job_name}-{i:02}' for i in range(N)] 
    df['jobname'] = jobnames
    df = move_df_col(df, 'jobname', 0)
    fname = Path(path, f"jobs.csv") 
    print(f'Saving to {str(fname)}')
    df.to_csv(fname, index=False) 


    # for flags that are shared across all jobs, we store in json
    jobargs = {
        'seed': args.seed,
        'dataset': args.dataset,
        'data_dim': args.data_dim,
        'dgp_type': args.dgp_type,
        'dgp_str': args.dgp_str, 
        'model_type': args.model_type,
        'ntrain': args.ntrain,
        'ntest': args.ntest,
        'ntrials': args.ntrials,
    }
    fname = Path(path, "args.json")
    with open(fname, 'w') as json_file:
        json.dump(jobargs, json_file, indent=4)

    cmd_dict = {}
    cmd_list = []
    for index, row in df.iterrows():
        for i in range(args.ntrials):
            jobname = f'{row.jobname}-trial{i}'
            cmd = make_unix_cmd_given_flags(
                row.algo, row.param, row.lr, row.niter, row.nsample, row.lin, row.ef, row.dlr_rank, 
                args.model_type, row.model_str, args.dataset, args.data_dim, 
                args.dgp_type, args.dgp_str, args.ntrain, args.ntest, args.seed + i)
            cmd_dict[jobname] = cmd
            cmd_list.append(cmd)
    #df['command'] = cmd_list
       
    
    cmds = [{'jobname': key, 'command': value} for key, value in cmd_dict.items()]
    df_cmds = pd.DataFrame(cmds)
    fname = Path(path, "cmds.csv") 
    df_cmds.to_csv(fname, index=False)


         
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--job_name", type=str)
    parser.add_argument("--seed", type=int,  default=0)
    parser.add_argument("--ntrials", type=int,  default=1)

    # Data parameters
    parser.add_argument("--dataset", type=str, default="reg")  
    parser.add_argument("--data_dim", type=int,  default=10)
    parser.add_argument("--dgp_type", type=str, default="lin") # or mlp
    parser.add_argument("--dgp_str", type=str, default="") # 20_20_1 
    parser.add_argument("--ntrain", type=int,  default=1000)
    parser.add_argument("--ntest", type=int,  default=1000)

    # Agent parameters
    parser.add_argument("--algo_list", type=str, nargs="+", default=["bong"])
    parser.add_argument("--param_list", type=str, nargs="+", default=["fc"])
    parser.add_argument("--lin_list", type=int, nargs="+", default=[0])
    parser.add_argument("--lr_list", type=float, nargs="+", default=[0.01])
    parser.add_argument("--niter_list", type=int, nargs="+", default=[10])
    parser.add_argument("--nsample_list", type=int, nargs="+", default=[100])
    parser.add_argument("--ef_list", type=int, nargs="+", default=[1])
    parser.add_argument("--rank_list", type=int, nargs="+", default=[10])

    # Model parameters
    parser.add_argument("--model_type", type=str, default="lin") # or mlp
    parser.add_argument("--model_str_list", type=str, nargs="+", default="")


    args = parser.parse_args()
    print(args)
    
    main(args)