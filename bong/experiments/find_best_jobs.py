
import argparse
import os
import itertools
import pandas as pd
from pathlib import Path
import os
import numpy as np



def main(args):
    results_dir = args.dir
 
    fname = f"{results_dir}/jobs_with_eval.csv"
    df = pd.read_csv(fname)

    #df['minscore'] = df.groupby('algo')[args.metric].transform('min')
    #condition = (df['minscore'] != df[args.metric])
    #indices_to_drop = df[condition].index
    #df_filtered = df.drop(indices_to_drop)
   

    grouped = df.groupby(['algo', 'param', 'lin', 'ef'])
    # Find the indices of the max score within each group
    idx = grouped[args.metric].idxmin()
    # Use these indices to get the corresponding rows
    #keep = ['jobname', 'algo', 'param', 'lin', 'ef', args.metric]
    #df_filtered = df.loc[idx, keep].reset_index(drop=True)
    df_filtered = df.loc[idx].reset_index(drop=True)
    
    fname = f"{results_dir}/best_jobs.csv"
    print(f'Writing to {fname}')
    df_filtered.to_csv(fname, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    #parser.add_argument("--metric", type=str, default="nlpd_val_mid")
    parser.add_argument("--metric", type=str, default="nlpd_te_final")

    args = parser.parse_args()
    main(args)