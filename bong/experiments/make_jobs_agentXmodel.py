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

def make_df_crossproduct(agent_list, model_list):
    args_list = []
    for agent in agent_list:
        for model in model_list:
            args = {'agent': agent, 'model_neurons_str': model}
            args_list.append(args)
    df = pd.DataFrame(args_list)
    df = df.drop_duplicates()
    return df

def main(args):
    results_dir = args.dir
    print(f'Saving jobs to {results_dir}')
    path = Path(results_dir)
    path.mkdir(parents=True, exist_ok=True)

    df = make_df_crossproduct(args.agent_list, args.model_neurons_str_list)
    #df['agent'] = args.agent
    df['lr'] = args.lr 
    df['niter'] = args.niter
    df['nsample'] = args.nsample
    df['linplugin'] = args.linplugin
    df['ef'] = args.ef
    df['dlr_rank'] = args.rank
    df['model_type'] = args.model_type
    #df['model_neurons_str'] = args.model_neurons_str
    df['dataset'] = args.dataset
    df['ntrain'] = args.ntrain
    df['data_dim'] = args.data_dim
    df['dgp_type'] = args.dgp_type
    df['dgp_neurons_str'] = args.dgp_neurons_str #make_neuron_str(args.dgp_neurons)
    
    N = len(df)
    jobnames = [f'{args.job_name}-{i:02}' for i in range(N)] 
    df['jobname'] = jobnames
    df = move_df_col(df, 'jobname', 0)

    print(df)

    cmd_dict = {}
    cmd_list = []
    for index, row in df.iterrows():
        cmd = make_unix_cmd_given_flags(
            row.agent, row.lr, row.niter, row.nsample,
            row.linplugin, row.ef, row.dlr_rank, 
            row.model_type, row.model_neurons_str, 
            row.dataset, row.data_dim, 
            row.dgp_type, row.dgp_neurons_str, row.ntrain)
        cmd_dict[row.jobname] = cmd
        cmd_list.append(cmd)
    df['command'] = cmd_list
       

    fname = Path(path, f"jobs.csv")
    print(f'Saving to {str(fname)}')
    df.to_csv(fname, index=False) 


   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--job_name", type=str)

    parser.add_argument("--dataset", type=str, default="reg") 
    parser.add_argument("--ntrain", type=int, default=500)
    parser.add_argument("--data_dim", type=int, default=10)
    parser.add_argument("--dgp_type", type=str, default="lin") # or mlp
    parser.add_argument("--dgp_neurons_str", type=str, default="") # 20_20_1 


    #parser.add_argument("--agent", type=str, default="bong_fc", choices=AGENT_NAMES)
    parser.add_argument("--agent_list", type=str, default="bong_fc", nargs="+")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--niter", type=int, default=10) 
    parser.add_argument("--nsample", type=int, default=100) 
    parser.add_argument("--ef", type=int, default=1)
    parser.add_argument("--linplugin", type=int, default=0)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--model_type", type=str, default="lin") # or mlp
    #parser.add_argument("--model_neurons_str", type=str, default="")
    parser.add_argument("--model_neurons_str_list", type=str, nargs="+", default="")

    args = parser.parse_args()
    print(args)
    
    main(args)