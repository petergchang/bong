
import argparse
import os
import itertools
import pandas as pd
from pathlib import Path
import os
import numpy as np



def main(args):
    results_dir = args.dir
    df_eval = pd.read_csv(f'{results_dir}/eval.csv')
    df_jobs = pd.read_csv(f'{results_dir}/jobs.csv')
    df_jobs = df_jobs.drop(columns=['command'])
    df = pd.merge(df_jobs, df_eval, on='jobname', how='inner')
    df['minscore'] = df.groupby('algo')[args.metric].transform('min')
    condition = (df['minscore'] != df[args.metric])
    indices_to_drop = df[condition].index
    df_filtered = df.drop(indices_to_drop)
   
    fname = f"{results_dir}/best_jobs.csv"
    print(f'Writing to {fname}')
    df_filtered.to_csv(fname, index=False)
    print(df_filtered)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--metric", type=str, default="nlpd_val_mid")

    args = parser.parse_args()
    main(args)