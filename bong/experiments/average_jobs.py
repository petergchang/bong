
import argparse
import os
import itertools
import pandas as pd
from pathlib import Path
import os
import datetime
import jax.numpy as jnp
import numpy as np
import json


from job_utils import extract_results_from_files, extract_metrics_from_files

def copy_jobargs_deprecated(results_dir, dst_dir, jobnames):
    src = f"{results_dir}/jobs/{job}-trial0/args.json"
    with open(src, 'r') as json_file:
        args = json.load(json_file)
    dst = f"{dst_dir}/args.json"
    args['metrics'] = list(metric_dict.keys())

    if 0:
        cmd = f'cp {src} {dst}'
        print(f'Running {cmd}')
        try:
            os.system(cmd)
        except Exception as e:
            print(f'Error {e}')


def process_metrics(results_dir, ntrials, job):
    metric_dict = {}
    metric_names = extract_metrics_from_files(results_dir, jobs_file="jobs.csv", jobs_suffix='-trial0', exclude_val=False)
    for metric in metric_names:
        data_list = []
        for i in range(ntrials):
            jobname = f'{job}-trial{i}'
            fname = f"{results_dir}/jobs/{jobname}/results.csv"
            if not os.path.isfile(fname):
                print(f'This file does not exist, skipping:', fname)
                continue
            df_res = pd.read_csv(fname)
            vals = df_res[metric].to_numpy()
            data_list.append(vals)
        data_mat = jnp.array(data_list)
        data_mean = jnp.mean(data_mat, axis=0)
        data_var = jnp.var(data_mat, axis=0)
        metric_dict[f'{metric}_mean'] = data_mean
        metric_dict[f'{metric}_var'] = data_var
    return metric_dict

def process_jobargs(results_dir, ntrials, job):
    elapsed_list = []
    summary_list = []
    seed_list = []
    for i in range(ntrials):
        fname = f"{results_dir}/jobs/{job}-trial{i}/args.json"
        if not os.path.isfile(fname):
            print(f'This file does not exist, skipping:', fname)
            continue
        with open(fname, 'r') as json_file:
            args = json.load(json_file)
        elapsed_list.append(args['elapsed'])
        summary_list.append(args['summary'])
        seed_list.append(args['seed'])

    elapsed_mat = np.array(elapsed_list)
    elapsed_mean = np.mean(elapsed_mat)
    elapsed_var = np.var(elapsed_mat)

    # Mutate the args dict from the last trial
    args['seed'] = seed_list
    args['summary'] = '\n'.join(summary_list)
    args['elapsed'] = None #elapsed_mean
    args['elapsed_mean'] = elapsed_mean 
    args['elapsed_var'] = elapsed_var
    return args 


def main(args):
    path = Path(args.dir)
    results_dir = str(path)
    fname = f"{results_dir}/jobs.csv"
    df = pd.read_csv(fname)
    jobnames = df['jobname']

    fname = f"{results_dir}/args.json"
    with open(fname, 'r') as json_file:
        jobargs = json.load(json_file)
    ntrials = jobargs['ntrials']
    print('ntrials', ntrials)

    for job in jobnames:
        dst_dir = f'{results_dir}/jobs/{job}-averaged'
        dst_path = Path(dst_dir)
        print(f'\n Creating {dst_dir}')
        dst_path.mkdir(parents=True, exist_ok=True)

        metric_dict = process_metrics(results_dir, ntrials, job)
        dst = f"{dst_dir}/results.csv"
        df_metrics = pd.DataFrame(metric_dict)
        df_metrics.to_csv(dst, index=False)

        jobargs = process_jobargs(results_dir, ntrials, job)
        dst = f"{dst_dir}/args.json"
        with open(dst, 'w') as json_file:
            json.dump(jobargs, json_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="")

    args = parser.parse_args()
    main(args)