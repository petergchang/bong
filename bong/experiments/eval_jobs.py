
import argparse
import os
import itertools
import pandas as pd
from pathlib import Path
import os
import numpy as np
import datetime
import json
from job_utils import extract_results_from_files

def replace_nan_with_infty(x):
    if np.isnan(x):
        return 1e20
    else:
        return x

def create_df(results_dir):
    nlpd_te_results = extract_results_from_files(results_dir,  'nlpd')
    nlpd_val_results = extract_results_from_files(results_dir,  'nlpd_val')
    jobnames = nlpd_te_results.keys()
    nlpd_te_final, nlpd_val_final = {}, {}
    nlpd_te_mid, nlpd_val_mid = {}, {}
    for jobname in jobnames:
        nlpd = nlpd_te_results[jobname]['vals'].to_numpy()
        nlpd_te_final[jobname] = replace_nan_with_infty(nlpd[-1])
        T = len(nlpd)
        mid = int(0.5*T) 
        nlpd_te_mid[jobname] = replace_nan_with_infty(nlpd[mid])

        nlpd = nlpd_val_results[jobname]['vals'].to_numpy()
        nlpd_val_final[jobname] = replace_nan_with_infty(nlpd[-1])
        T = len(nlpd)
        mid = int(0.5*T) 
        nlpd_val_mid[jobname] = replace_nan_with_infty(nlpd[mid])

    dicts = {
        'nlpd_te_final': nlpd_te_final,
        'nlpd_te_mid': nlpd_te_mid,
        'nlpd_val_final': nlpd_val_final,
        'nlpd_val_mid': nlpd_val_mid,
    }
    data = {'jobname': jobnames}
    for label, d in dicts.items():
        data[label] = [d[key] for key in jobnames]
    df = pd.DataFrame(data)
    return df


def main(dirname):
    fname = f"{dirname}/eval.csv"
    print(f'Writing to {fname}')
    df = create_df(dirname)
    df.to_csv(fname, index=False)
    print(df)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)

    args = parser.parse_args()
    main(args.dir)