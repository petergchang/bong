import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from pathlib import Path
import os
import matplotlib.image as mpimg
import json
from bong.util import find_first_true
from bong.agents import make_agent_args

cwd = Path(os.getcwd())
root = cwd
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

def make_neuron_str(neurons):
    s = [str(n) for n in neurons]
    neurons_str = "_".join(s)
    return neurons_str

def parse_neuron_str(s):
    neurons_str = s.split("_")
    neurons = [int(n) for n in neurons_str]
    return neurons



def make_unix_cmd_given_flags(algo, param, lr, niter, nsample, linplugin, ef, rank,
                            model_type, model_str,
                            dataset, data_dim, dgp_type, dgp_str, ntrain):
    # We must pass in all flags where we want to override the default in run_job
    #main_name = '/teamspace/studios/this_studio/bong/bong/experiments/run_job.py'
    main_name = f'{script_dir}/run_job.py'
    #model_neurons = unmake_neuron_str(model_neurons_str)
    #dgp_neurons = unmake_neuron_str(dgp_neurons_str)
    cmd = (
        f'python {main_name} --algo {algo} --param {param} --lr {lr}'
        f' --niter {niter} --nsample {nsample} --lin {linplugin}'
        f' --ef {ef} --rank {rank}'
        f' --model_type {model_type} --model_str {model_str}'
        f' --dataset {dataset} --data_dim {data_dim}'
        f' --dgp_type {dgp_type} --dgp_str {dgp_str}'
        f' --ntrain {ntrain}'
    )
    return cmd

def make_df_crossproduct( 
        algo_list, param_list, lin_list,
        lr_list, niter_list, nsample_list, ef_list, rank_list, model_str_list):
    args_list = []
    all_args_list = []
    for algo in algo_list:
        for param in param_list:
            for lin in lin_list:
                for lr in lr_list:
                    for niter in niter_list:
                        for nsample in nsample_list:
                            for ef in ef_list:
                                for rank in rank_list:
                                    for model_str in model_str_list:
                                        all_args = {'algo': algo, 'param': param, 'lin': lin, 'lr': lr, 
                                                'niter': niter, 'nsample': nsample,  'ef': ef, 'dlr_rank': rank, 
                                                'model_str': model_str}
                                        all_args_list.append(all_args)        
                                        
                                        if (param == 'dlr') and (lin==0) and (ef == 0): continue # sanpled Hessians not implemented for DLR

                                        args = make_agent_args(algo, param, lin, rank, ef, nsample, niter, lr)
                                        args['model_str'] = model_str
                                        args_list.append(args)
    #df = pd.DataFrame(all_args_list)
    df = pd.DataFrame(args_list)
    df_unique = df.drop_duplicates(df)
    df_unique = df_unique.reset_index(drop=True)
    return df_unique


def extract_metrics_from_files(dir, exclude_val=True, jobs_file="jobs.csv"):
    fname = f"{dir}/{jobs_file}"
    df = pd.read_csv(fname)
    jobnames = df['jobname']
    jobname = jobnames[0]
    fname = f"{dir}/jobs/{jobname}/results.csv"
    df_res = pd.read_csv(fname)
    metrics =  df_res.columns
    if exclude_val:
        metrics = [m for m in metrics if "_val" not in m ]
    return metrics




def extract_results_from_files(dir,  metric, jobs_file="jobs.csv"):
    fname = f"{dir}/{jobs_file}"
    df = pd.read_csv(fname)
    jobnames = df['jobname']
    results = {}
    for jobname in jobnames:
        fname = f"{dir}/jobs/{jobname}/results.csv"
        if not os.path.isfile(fname):
            print(f'This file does not exist, skipping:', fname)
            continue
        df_res = pd.read_csv(fname)
        vals = df_res[metric].to_numpy()
        nans = jnp.isnan(vals)
        if jnp.any(nans):
            T = find_first_true(nans)
        else:
            T = len(vals)
        
        fname = f"{dir}/jobs/{jobname}/args.json"
        with open(fname, 'r') as json_file:
            args = json.load(json_file)
        d = {
            'metric': metric,
            'vals': vals,
            'valid_len': T,
            'agent_name': args['agent_name'],
            'agent_full_name': args['agent_full_name'],
            'model_name': args['model_name'],
            'data_name': args['data_name'],
            'elapsed': args['elapsed'],
            }
        results[jobname] = d
    return results

