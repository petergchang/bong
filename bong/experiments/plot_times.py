import argparse
import os
import itertools
import pandas as pd
from pathlib import Path
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import json

from job_utils import extract_results_from_files, extract_metrics_from_files
from bong.util import make_file_with_timestamp, parse_full_name, make_full_name


def make_plot_params(algo, ef, lin):
    markers = {'bong': 'o', 'blr': 's', 'bog': 'x', 'bbb': '*'}
    marker = markers[algo]
    if (ef==0) & (lin==0):
        linestyle = '-'
    elif (ef==1) & (lin==0): 
        linestyle = '--'
    else:
        linestyle = ':' # lin==1
    return {
            'linestyle': linestyle,
            'linewidth': 2,
            'marker': marker,
            'markersize': 10
            }


def main(args):
    results_dir = args.dir
    fname = f"{results_dir}/jobs.csv"
    df = pd.read_csv(fname)
    jobnames = df['jobname']
    pt_dict_per_agent = {} #(param, time) pair per agent
    agent_name_list = []
    full_agent_name_dict = {}
    for i, jobname in enumerate(jobnames):
        fname = f"{results_dir}/{jobname}/work/args.json"
        with open(fname, 'r') as json_file:
            res = json.load(json_file)

        full_name = res['agent_full_name']
        parts = parse_full_name(full_name)
        algo, param, lin, ef, rank = parts['algo'], parts['param'], parts['lin'], parts['ef'], parts['rank']   
        agent_name = f'{algo}_{param}_Lin{lin}_EF{ef}_R{rank}'
        agent_name_list.append(agent_name)
        full_agent_name_dict[agent_name] = full_name 
        nparams = res['model_nparams']
        T = res['ntrain']
        elapsed = res['elapsed']
        steptime = elapsed/T
        if agent_name in pt_dict_per_agent:
            times_per_param = pt_dict_per_agent[agent_name]
            times_per_param[nparams] = steptime
            pt_dict_per_agent[agent_name] = times_per_param
        else:
            times_per_param = {nparams: steptime}
            pt_dict_per_agent[agent_name] = times_per_param

    agent_names = full_agent_name_dict.keys()
    nparams_per_agent = {}
    times_per_agent = {}
    for agent in agent_names:
        times_per_param = pt_dict_per_agent[agent]
        nparams_per_agent[agent] = np.array(list(times_per_param.keys()))
        times_per_agent[agent] = np.array(list(times_per_param.values()))
    

    fig, ax = plt.subplots(figsize=(8,6)) # width, height in inches
    fname = f"{results_dir}/times"
    print(f'Saving figure to {fname}')
    for agent in agent_names:
        agent_name_long = full_agent_name_dict[agent]
        parts = parse_full_name(agent_name_long)
        algo, param, lin, ef = parts['algo'], parts['param'], parts['lin'], parts['ef']
        kwargs = make_plot_params(algo, ef, lin)
        ax.plot(nparams_per_agent[agent], times_per_agent[agent], label=agent_name_long, **kwargs)
    ax.grid()
    ax.legend()
    ax.set_ylabel("Elapsed time per step (sec)")
    ax.set_xlabel("Num. parameters")
    fig.savefig(f'{fname}.png', bbox_inches='tight', dpi=300)
    fig.savefig(f'{fname}.pdf', bbox_inches='tight', dpi=300)

    return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="")

    args = parser.parse_args()
    main(args)