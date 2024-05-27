
import argparse
import os
import itertools
import pandas as pd
from pathlib import Path
import os
import numpy as np
import datetime
import json
from job_utils import extract_results

def replace_nan_with_infty(x):
    if np.isnan(x):
        return 1e20
    else:
        return x

def create_eval_df(args):
    nll_val_results = extract_results(args.dir,  'nlpd-pi_val_mean', args.jobs_file, args.jobs_suffix)
    nlpd_val_results = extract_results(args.dir,  'nlpd-mc_val_mean', args.jobs_file, args.jobs_suffix)
    jobnames = nlpd_val_results.keys()
    nll_val_final, nlpd_val_final, nll_val_mid, nlpd_val_mid = {}, {}, {}, {}
    for jobname in jobnames:
        nlpd = nlpd_val_results[jobname]
        nlpd_val_final[jobname] = replace_nan_with_infty(nlpd[-1])
        T = len(nlpd)
        mid = int(0.5*T) 
        nlpd_val_mid[jobname] = replace_nan_with_infty(nlpd[mid])

        nll = nll_val_results[jobname]
        nll_val_final[jobname] = replace_nan_with_infty(nll[-1])
        T = len(nlpd)
        mid = int(0.5*T) 
        nll_val_mid[jobname] = replace_nan_with_infty(nll[mid])

    dicts = {
        'nlpd-mc_val_final': nlpd_val_final,
        'nlpd-mc_val_mid': nlpd_val_mid,
        'nlpd-pi_val_final': nll_val_final,
        'nlpd-pi_val_mid': nll_val_mid,
    }
    data = {'jobname': jobnames}
    for label, d in dicts.items():
        data[label] = [d[key] for key in jobnames]
    df = pd.DataFrame(data)
    return df

def create_merged_df(args):
    df_eval = pd.read_csv(f'{args.dir}/eval.csv')
    df_jobs = pd.read_csv(f'{args.dir}/{args.jobs_file}')
    #print(df_jobs.columns)
    #df_jobs = df_jobs.drop(columns=['command'])
    #df_jobs = df_jobs.drop(columns=['dataset', 'data_dim', 'dgp_type', 'dgp_str', 'ntrain'])
    #df_jobs  = df_jobs.drop(columns=['model_type'])
    df = pd.merge(df_jobs, df_eval, on='jobname', how='inner')
    return df


def main(args):
    fname = f"{args.dir}/eval.csv"
    print(f'Writing to {fname}')
    df_eval = create_eval_df(args)
    df_eval.to_csv(fname, index=False)

    fname = f"{args.dir}/jobs_with_eval.csv"
    print(f'Writing to {fname}')
    df_merged = create_merged_df(args)
    df_merged.to_csv(fname, index=False)
    
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--jobs_file", type=str, default="jobs.csv")
    parser.add_argument("--jobs_suffix", type=str, default="-averaged")

    args = parser.parse_args()
    main(args)