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
from bong.agents import parse_agent_full_name, make_agent_name_from_parts


def plot_times_old(results):
    jobnames = results.keys()
    fig, ax = plt.subplots(figsize=(8,6)) # width, height in inches
    times = {}
    for i, jobname in enumerate(jobnames):
        res = results[jobname]
        vals = res['vals']
        expt_name = res['agent_name']
        parts = parse_agent_full_name(expt_name)
        short_name = make_agent_name_from_parts(parts['algo'], parts['param'], parts['lin'])
        T = len(vals)
        elapsed = 1000*round(res['elapsed']/T, 3)
        times[short_name] = elapsed # we assume each agent only occurs once

    names, vals = times.keys(), times.values()
    ax.bar(names, vals)
    ax.set_ylabel('time (ms) per step')
    ax.set_title('Running time (in ms) per step', fontsize=12)
    ax.set_xticks(range(len(names)))  # Set the positions of the ticks
    ax.set_xticklabels(names, rotation=45, fontsize=10) 

    return fig, ax

def main(args):
    results_dir = args.dir

    fname = f"{results_dir}/jobs.csv"
    df = pd.read_csv(fname)
    jobnames = df['jobname']
    pt_dict_per_agent = {}
    agent_name_list = []
    for i, jobname in enumerate(jobnames):
        fname = f"{results_dir}/{jobname}/work/args.json"
        with open(fname, 'r') as json_file:
            res = json.load(json_file)
        agent_name_long = res['agent_name']
        parts = parse_agent_full_name(agent_name_long)
        algo, param, lin, ef = parts['algo'], parts['param'], parts['lin'], parts['ef']
        agent_name = f'{algo}_{param}_Lin{lin}_EF{ef}'
        agent_name_list.append(agent_name)
        model_name = res['model_name']
        nparams = res['model_nparams']
        T = res['ntrain']
        elapsed = res['elapsed']
        if agent_name in pt_dict_per_agent:
            times_per_param = pt_dict_per_agent[agent_name]
            times_per_param[nparams] = elapsed
            pt_dict_per_agent[agent_name] = times_per_param
        else:
            times_per_param = {nparams: elapsed}
            pt_dict_per_agent[agent_name] = times_per_param
    
    agent_names = set(agent_name_list)
    params_per_agent = {}
    times_per_agent = {}
    for agent in agent_names:
        times_per_param = pt_dict_per_agent[agent]
        params_per_agent[agent] = np.array(list(times_per_param.keys()))
        times_per_agent[agent] = np.array(list(times_per_param.values()))
    

    fig, ax = plt.subplots(figsize=(8,6)) # width, height in inches
    fname = f"{results_dir}/times"
    print(f'Saving figure to {fname}')
    for agent in agent_names:
        print(agent, params_per_agent[agent], times_per_agent[agent])
        ax.plot(params_per_agent[agent], times_per_agent[agent], 'o-', label=agent)
    ax.grid()
    ax.legend()
    ax.set_ylabel("Elapsed time (sec)")
    ax.set_xlabel("Num. parameters")
    fig.savefig(f'{fname}.png', bbox_inches='tight', dpi=300)
    fig.savefig(f'{fname}.pdf', bbox_inches='tight', dpi=300)

    return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="")

    args = parser.parse_args()
    main(args)