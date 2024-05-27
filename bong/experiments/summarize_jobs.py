
import argparse
import os
import itertools
import pandas as pd
from pathlib import Path
import os
import datetime
import json



def create_summary_df(args):
    fname = f"{args.dir}/{args.jobs_file}"
    print(f'Reading from {fname}')
    df = pd.read_csv(fname)
    jobnames = df['jobname']

    # Create dict of dicts, containing summary results for each experiment
    meta = {}
    for jobname in jobnames:
        fname = f"{args.dir}/jobs/{jobname}{args.jobs_suffix}/args.json"
        if not os.path.isfile(fname):
            print(f'This file does not exist, skipping:', fname)
            continue
        with open(fname, 'r') as json_file:
            sub = json.load(json_file)
            #meta[jobname] = args
            keep = {'agent_name', 'model_name',  'elapsed_mean', 'summary'}
            #keep = {'agent_name',  'elapsed', 'summary'}
            d = {}
            for k in keep:
                d[k] = sub[k]
            meta[jobname] = d

    # Merge dict of dicts into a dataframe
    df = pd.DataFrame()
    # Iterate over the outer dictionary and create a DataFrame for each nested dictionary
    for key, value in meta.items():
        temp_df = pd.DataFrame([value])
        temp_df['jobname'] = key
        df = pd.concat([df, temp_df], ignore_index=True)
    #df = df.drop(columns=['dir'])
    # Reorder the columns if necessary
    #df = df[['jobname', 'agent_name', 'model_name', 'data_name', 'elapsed', 'summary']]
    df = df[['jobname', 'agent_name', 'model_name', 'elapsed_mean', 'summary']]

    return df

def main(args):
    fname = f"{args.dir}/summary.csv"
    print(f'Writing to {fname}')
    df = create_summary_df(args)
    df.to_csv(fname, index=False)
    print(df)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--jobs_file", type=str, default="jobs.csv")
    parser.add_argument("--jobs_suffix", type=str, default="-averaged")

    args = parser.parse_args()
    main(args)