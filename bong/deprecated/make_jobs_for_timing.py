import argparse
import os
import itertools
import pandas as pd
from pathlib import Path
import os
import datetime

from bong.util import  move_df_col
from job_utils import make_unix_cmd_given_flags, make_df_crossproduct

def main(args):
    results_dir = args.dir
    print(f'Saving jobs to {results_dir}')
    path = Path(results_dir)
    path.mkdir(parents=True, exist_ok=True)


    df = make_df_crossproduct(
        args.algo_list, args.param_list, args.lin_list,
        [args.lr], [args.niter], [args.nsample],
        args.ef_list, args.rank_list, args.model_str_list, [args.key])
    
    df['model_type'] = args.model_type
    df['dataset'] = args.dataset
    df['ntrain'] = args.ntrain
    df['ntest'] = args.nest
    df['data_dim'] = args.data_dim
    df['dgp_type'] = args.dgp_type
    df['dgp_str'] = args.dgp_str
    
    N = len(df)
    jobnames = [f'{args.job_name}-{i:02}' for i in range(N)] 
    df['jobname'] = jobnames
    df = move_df_col(df, 'jobname', 0)

    fname = Path(path, f"jobs.csv")
    print(f'Saving to {str(fname)}')
    df.to_csv(fname, index=False) 

    cmd_dict = {}
    cmd_list = []
    for index, row in df.iterrows():
        cmd = make_unix_cmd_given_flags(
            row.algo, row.param, row.lr, row.niter, row.nsample,
            row.lin, row.ef, row.dlr_rank, 
            row.model_type, row.model_str, 
            row.dataset, row.data_dim, 
            row.dgp_type, row.dgp_str, row.ntrain, row.ntest, row.key)
        cmd_dict[row.jobname] = cmd
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
    parser.add_argument("--key", type=int, default=0)

    parser.add_argument("--dataset", type=str, default="reg") 
    parser.add_argument("--ntrain", type=int, default=500)
    parser.add_argument("--ntest", type=int, default=1000)
    parser.add_argument("--data_dim", type=int, default=10)
    parser.add_argument("--dgp_type", type=str, default="lin") # or mlp
    parser.add_argument("--dgp_str", type=str, default="") # 20_20_1 


    parser.add_argument("--algo_list", type=str, default=["bong"], nargs="+")
    parser.add_argument("--param_list", type=str, default=["fc"], nargs="+")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--niter", type=int, default=10) 
    parser.add_argument("--nsample", type=int, default=100) 
    parser.add_argument("--ef_list", type=int, nargs="+", default=[1])
    parser.add_argument("--lin_list", type=int, nargs="+", default=[0])
    parser.add_argument("--rank_list", type=int, nargs="+", default=[0])
    parser.add_argument("--model_type", type=str, default="lin") # or mlp
    parser.add_argument("--model_str_list", type=str, nargs="+", default="")

    args = parser.parse_args()
    print(args)
    
    main(args)