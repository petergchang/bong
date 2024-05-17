import argparse
import os
import itertools
import pandas as pd
from pathlib import Path
import os
import datetime

from bong.agents import AGENT_DICT, AGENT_NAMES, make_agent_name_from_parts, extract_optional_agent_args
from bong.util import safestr, make_neuron_str, unmake_neuron_str, make_file_with_timestamp, move_df_col
from job_utils import make_unix_cmd_given_flags

def make_df_crossproduct( 
        algo_list, param_list, lin_list,
        lr_list, niter_list, nsample_list, ef_list, rank_list):
    args_list = []
    for algo in algo_list:
        for param in param_list:
            for lin in lin_list:
                agent = make_agent_name_from_parts(algo, param, lin)
                props = AGENT_DICT[agent]
                for lr in lr_list:
                    for niter in niter_list:
                        for nsample in nsample_list:
                            for ef in ef_list:
                                for rank in rank_list:
                                    args = extract_optional_agent_args(props, lr, niter, nsample, ef, rank)
                                    args['agent'] = agent
                                    args_list.append(args)
    df = pd.DataFrame(args_list)
    df = df.drop_duplicates()
    return df




def main(args):
    path = Path(args.dir)
    results_dir = str(path)
    print(f'Saving jobs to {results_dir}')
    path.mkdir(parents=True, exist_ok=True)

    df_flags = make_df_crossproduct(
        args.algo_list, args.param_list, args.lin_list,
        args.lr_list, args.niter_list, args.nsample_list,
        args.ef_list, args.rank_list)

    # for flags that are shared across all jobs, we create extra columns (duplicated across rows)
    df_flags['dataset'] = args.dataset
    df_flags['data_dim'] = args.data_dim
    df_flags['dgp_type'] = args.dgp_type
    df_flags['dgp_neurons_str'] = args.dgp_neurons_str # make_neuron_str(args.dgp_neurons)
    df_flags['model_type'] = args.model_type
    df_flags['model_neurons_str'] = args.model_neurons_str #make_neuron_str(args.model_neurons)
    df_flags['ntrain'] = args.ntrain

    N = len(df_flags)
    jobnames = [f'{args.job_name}-{i:02}' for i in range(N)] 
    df_flags['jobname'] = jobnames
    df_flags = move_df_col(df_flags, 'jobname', 0)

    cmd_dict = {}
    cmd_list = []
    for index, row in df_flags.iterrows():
        cmd = make_unix_cmd_given_flags(
            row.agent, row.lr, row.niter, row.nsample,
            row.linplugin, row.ef, row.dlr_rank, 
            row.model_type, row.model_neurons_str, 
            row.dataset, row.data_dim, 
            row.dgp_type, row.dgp_neurons_str, row.ntrain)
        cmd_dict[row.jobname] = cmd
        cmd_list.append(cmd)
    df_flags['command'] = cmd_list
       

    fname = Path(path, f"jobs.csv")
    print(f'Saving to {str(fname)}')
    df_flags.to_csv(fname, index=False) 


    #cmds = [{'jobname': key, 'command': value} for key, value in cmd_dict.items()]
    #df_cmds = pd.DataFrame(cmds)
    #fname = Path(path, "cmds.csv")
    #df_cmds.to_csv(fname, index=False)
 
         
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--job_name", type=str)

    # Data parameters
    parser.add_argument("--dataset", type=str, default="reg")  
    parser.add_argument("--data_dim", type=int,  default=10)
    parser.add_argument("--dgp_type", type=str, default="lin") # or mlp
    #parser.add_argument("--dgp_neurons", type=int, nargs="+", default=[20,20,1]) 
    parser.add_argument("--dgp_neurons_str", type=str, default="") # 20_20_1 
    parser.add_argument("--ntrain", type=int,  default=500)
    
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
    #parser.add_argument("--model_neurons", type=int, nargs="+", default=[10, 10, 1])
    parser.add_argument("--model_neurons_str", type=str, default="") # 20_20_1 

    args = parser.parse_args()
    print(args)
    
    main(args)