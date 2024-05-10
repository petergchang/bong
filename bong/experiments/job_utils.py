from functools import partial
from typing import Any, Callable, Sequence
import argparse
from functools import partial
from pathlib import Path
import time
import re
import os
import json 
import pandas as pd
import jax
import jax.random as jr
import numpy as np
import jax.numpy as jnp

from bong.util import run_rebayes_algorithm
from bong.agents import AGENT_DICT, AGENT_NAMES


def make_results_dirname(jobname, parallel):
    # the results artefacts are written (by lightning ai studio) to a certain directory
    # depending on jobname.
    if parallel:
        output_dir = f'/teamspace/jobs/{jobname}/work'
    else:
        output_dir = f'/teamspace/studios/this_studio/jobs/{jobname}/work'
    return output_dir

def make_dict_for_flags(props, learning_rate, num_iter, num_sample):
    args = props.copy()
    if props['lr'] is None: args['lr'] = learning_rate
    if props['niter'] is None: args['niter'] = int(num_iter)
    if props['nsample'] is None: args['nsample'] = int(num_sample)
    args['ef'] = props['ef'] 
    args['linplugin'] = props['linplugin']
    del args['constructor']
    return args


def make_df_for_flag_crossproduct(agents, lrs, niters, nsamples, parallel):
    args_list = []
    for agent in agents:
        props = AGENT_DICT[agent]
        for lr in lrs:
            for niter in niters:
                for nsample in nsamples:
                    args = make_dict_for_flags(props, lr, niter, nsample)
                    args['agent'] = agent
                    args_list.append(args)
    df = pd.DataFrame(args_list)
    df = df.drop_duplicates()
    N = len(df)
    jobnames = [f'job-{i}' for i in range(N)] 
    df['jobname'] = jobnames
    dirs = [make_results_dirname(j, parallel) for j in jobnames]
    df['results_dir'] = dirs
    return df

def make_unix_cmd_given_flags(agent, lr, niter, nsample):
    main_name = '/teamspace/studios/this_studio/bong/bong/experiments/run_one_job.py'
    cmd = f'python {main_name} --agent {agent} --lr {lr} --niter {niter} --nsample {nsample}'
    return cmd


def make_cmd_dict_for_flag_crossproduct(args):
    df_flags = make_df_for_flag_crossproduct(args.agents, args.lrs, args.niters, args.nsamples, args.parallel)
    df_flags['dataset'] = args.dataset

    cmd_dict = {}
    for index, row in df_flags.iterrows():
        cmd = make_unix_cmd_given_flags(row.agent, row.lr, row.niter, row.nsample)
        cmd_dict[row.jobname] = cmd

    # Store csv containing all the commands that are being executed
    path = Path(args.dir)
    path.mkdir(parents=True, exist_ok=True)
    fname = Path(path, "flags.csv")
    print("Saving to", fname)
    df_flags.to_csv(fname, index=False) 

    cmds = [{'agent': key, 'command': value} for key, value in cmd_dict.items()]
    df_cmds = pd.DataFrame(cmds)
    fname = Path(path, "cmds.csv")
    print("Saving to", fname)
    df_cmds.to_csv(fname, index=False)

    return cmd_dict


#    grid_search_params = list(itertools.product(args.agent, args.learning_rate, args.num_iter))
    #grid_search_params = [(lr, agent) for lr in args.learning_rate for agent in args.agent]
    #for index, (agent, lr, niter) in enumerate(grid_search_params):
    #    cmd = make_cmd(agent, lr, niter)
    #    job_name = f'bong-{index}'
    #    output_dir = f'/teamspace/studios/this_studio/jobs/{job_name}/work'
    #    cmd = cmd + f' --dir {output_dir}'
    #    print('running', cmd)
    #    os.system(cmd)

def run_agent(key, agent, data, callback):
    print(f"Running {agent.name}")
    t0 = time.perf_counter()
    _, (kldiv, nll, nlpd) = jax.block_until_ready(
        run_rebayes_algorithm(key, agent, data['X_tr'], data['Y_tr'], transform=callback)
    )
    t1 = time.perf_counter()
    print(f"KL-Div: {kldiv[-1]:.4f}, NLL: {nll[-1]:.4f},  NLPD: {nlpd[-1]:.4f}, Time: {t1 - t0:.2f}s")
    ntest = len(kldiv)
    results = {
        'agent_name': agent.name,
        'dataset_name': data['name'],
        'time': t1 - t0, 
        'kl': kldiv, 
        'nll': nll,
        'nlpd': nlpd, 
        #'ntest': ntest
             }
    return results

def run_agents(keyroot, agent_queue, data, callback, result_dict={}):
    key = keyroot
    for agent_name, agent in agent_queue.items():
        results = run_agent(key, agent, data, callback)
        # store in legacy tuple format
        result_dict[agent_name] = (results['time'], results['kl'], results['nll'], results['nlpd'])
        keyroot, key = jr.split(keyroot)
    ntest = len(results['kl'])
    result_dict['ntest'] = ntest
    return result_dict

def run_agents_old(subkey, agent_queue, data, callback, result_dict={}):
    for agent_name, agent in agent_queue.items():
        print(f"Running {agent_name}...")
        key, subkey = jr.split(subkey)
        t0 = time.perf_counter()
        _, (kldiv, nll, nlpd) = jax.block_until_ready(
            run_rebayes_algorithm(key, agent, data['X_tr'], data['Y_tr'], transform=callback)
        )
        t1 = time.perf_counter()
        result_dict[agent_name] = (t1 - t0, kldiv, nll, nlpd)
        print(f"KL-Div: {kldiv[-1]:.4f}, NLL: {nll[-1]:.4f},  NLPD: {nlpd[-1]:.4f}, Time: {t1 - t0:.2f}s")
    ntest = len(kldiv)
    result_dict['ntest'] = ntest
    return result_dict

def list_subdirectories(directory):
    return [name for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))]

def get_job_dir(parallel):
    # for lightning ai studio
    # https://lightning.ai/docs/overview/train-models/hyperparameter-sweeps
    # Results are generated by experiments/run_jobs
    if not(parallel):
            data_dir = '/teamspace/studios/this_studio/jobs' 
    else:
            data_dir = '/teamspace/jobs/' 
    return data_dir
 

def read_job_args(parallel):
    '''Read args.json for each run and return a dataframe'''
    jobdir = get_job_dir(parallel)
    subdirs = list_subdirectories(jobdir)

    dicts = []
    for jobname in subdirs:
        path = Path(jobdir, jobname, 'work', 'args.json')
        if not(path.exists()):
            print('skipping ', path)
            continue

        # the with context manager fails inside a notebook
        #with open(path, 'r') as file:
        # args_dict = json.load(file)
        
        file = open(path, 'r')
        args_dict = json.load(file)
        file.close()
        args_dict['job-name'] = jobname
        args_dict.pop('dir')
        args_dict.pop('filename')

        dicts.append(args_dict)

    jobs_df = pd.DataFrame(dicts)
    return jobs_df


def read_job_results(parallel):
    '''Read results.csv for each run and return a dict of dataframes'''
    jobdir = get_job_dir(parallel)
    subdirs = list_subdirectories(jobdir)
    results_dict = {} # map job name to dataframe of results
    for jobname in subdirs:
        path = Path(jobdir, jobname, 'work', 'results.csv')
        if not(path.exists()):
            print('skipping ', path)
            continue
        # the with context manager fails inside a notebook
        #with open(path, 'r') as file:
        # args_dict = json.load(file)
        file = open(path, 'r')
        df = pd.read_csv(path)
        file.close()
        results_dict[jobname] = df
    return results_dict


def get_job_name(args_df, query_dict):
    query_str = ' & '.join([f"{k} == {repr(v)}" for k, v in query_dict.items()])
    #query_str = ' & '.join([f"{k}=={v}" for k, v in query_dict.items()])
    filtered_df = args_df.query(query_str)
    if len(filtered_df) > 1:
        msg= f'query is not unique, {query_str} matches {len(filtered_df)}'
        raise Exception(msg)
    jobname = filtered_df['job-name'].item()
    return jobname

def get_job_args(args_df, jobname):
    my_args = args_df[ args_df['job-name']==jobname ]
    return my_args

def test_jobs():
    parallel = True
    args_df = read_job_args(parallel)
    print(args_df)

    query_dict = {'agent': 'fg-bong', 'learning_rate': 2.0, 'num_iter': 10}
    jobname = get_job_name(args_df, query_dict)
    print(jobname)

    results_dict = read_job_results(parallel)
    print(results_dict.keys())
    print(results_dict[jobname])


def parse_filename(s):
    # example filename: "fg-l-bong-M10-I10-LR0_01"
    s = s.replace("_", ".")
    import re
    pattern = r"^([a-z-]+)-M(\d+)-I(\d+)-LR(\d*\.?\d+)"
    match = re.match(pattern, s)
    if match:
        return {
            'prefix': match.group(1),
            'M': int(match.group(2)),
            'I': int(match.group(3)),
            'LR': float(match.group(4))
        }
    return {'prefix': None, 'M': None, 'I': None, 'LR': None}

def test_parse_filename():
    strings = ["fg-bong-M10-I1-LR0", "fg-l-bong-M10-I10-LR0_01"]
    for string in strings:
        res = parse_filename(string)
        print(res)

def split_filename_column(df):
    # If filename is fg-bong-M10-I1-LR0_01, we create columns name, M, I, LR with corresponding values

    # Apply the parse function and expand the results into new DataFrame columns
    df_expanded = df['name'].apply(parse_filename).apply(pd.Series)

    # Join the new columns with the original DataFrame
    #df_final = df_expanded.join(df.drop('name', axis=1))
    df_final = df_expanded.join(df)

    # Optionally, rearrange columns to match the desired output format
    #df_final = df_final[['prefix', 'M', 'I', 'LR', 'step', 'kl', 'nll', 'nlpd', 'time']]
    return df_final



def extract_nsteps_from_result_dict(result_dict):
    # this breaks for laplace entry, which is a scalar, not a timeseries
    names = list(result_dict.keys())
    r = result_dict[names[0]]
    (tyme, kldiv, nll, nlpd) = r
    T = len(kldiv)
    return T

def convert_result_dict_to_pandas(result_dict):
    result_dict = result_dict.copy()
    #T = extract_nsteps_from_result_dict(result_dict)
    T = result_dict.pop('ntest')
    steps = range(0, T)

    if "laplace" in result_dict:
        laplace = result_dict.pop("laplace")
        (tim, kldiv, nll, nlpd) = laplace
        df  = pd.DataFrame({'name': 'laplace-M0-I0-LR0',  
                                'step': np.array([T]),
                                'kl': np.array(kldiv), 
                                'nll': np.array(nll), 
                                'nlpd': np.array(nlpd),
                                'time': tim,
                                })
        frames = [df]
    else:
        frames = []

    for name, r in result_dict.items():
        df  = pd.DataFrame({'name': name,  
                            'step': steps,
                            'kl': np.array(r[1]), 
                            'nll': np.array(r[2]), 
                            'nlpd': np.array(r[3]),
                            'time': r[0],
                            })
        frames.append(df)
    tbl = pd.concat(frames, ignore_index=True) # ensure each row has unique index
    return tbl

