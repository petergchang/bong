import argparse
import os
import itertools
import pandas as pd
from pathlib import Path
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import json

import jax.random as jr
import jax
import jax.numpy as jnp

from job_utils import extract_results, extract_jobargs, extract_metrics_from_files
from bong.util import make_file_with_timestamp, parse_full_name, make_full_name, parse_full_name_old
from bong.util import add_jitter, convolve_smooth, make_plot_params
from bong.util import find_first_true
from datasets import make_dataset
from models import fit_linreg_baseline, calc_mse, nll_linreg

def filter_jobnames(jobargs, exclude='', include=''):
    jobnames = list(jobargs.keys())
    keep_names = []
    for jobname in jobnames:
        res = jobargs[jobname]
        agent_name = res['agent_name']
        full_name = res['agent_full_name']
        parts = parse_full_name(full_name)
        algo, ef, lin, niter = parts['algo'], parts['ef'], parts['lin'], parts['niter']
        if len(exclude) > 1:
            skip_result = eval(exclude)
            if skip_result: 
                #print(f'Excluding {exclude}', res)
                continue
        if len(include) > 1:
            keep_result = eval(include)
            if not keep_result:
                #print(f'Only including {include}, so skipping', res)
                continue
        keep_names.append(jobname)
    return keep_names

def compute_linreg_baseline(results_dir):
    fname = f'{results_dir}/args.json'
    with open(fname, 'r') as json_file:
        jobargs = json.load(json_file)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=jobargs['dataset']) 
    parser.add_argument("--seed", type=int, default=jobargs['seed'])
    parser.add_argument("--ntrain", type=int, default=jobargs['ntrain'])
    parser.add_argument("--nval", type=int, default=1000)
    parser.add_argument("--ntest", type=int, default=jobargs['ntest'])
    parser.add_argument("--add_ones", type=int, default=0)
    args = parser.parse_args([])

    key = jr.PRNGKey(args.seed)
    key, subkey = jr.split(key)
    key, data = make_dataset(subkey, args)

    w, sigma2 = fit_linreg_baseline(data['X_tr'], data['Y_tr'], method='lstsq')
    prediction = data['X_te'] @ w 
    mse_linreg_baseline = calc_mse(prediction, data['Y_te'])
    obs_var = 0.1*jnp.var(data['Y_tr']) # same as used by agents
    nll_linreg_baseline = jnp.mean(jax.vmap(nll_linreg, (None, None, 0, 0))(w, obs_var, data['X_te'], data['Y_te']))

    return mse_linreg_baseline, nll_linreg_baseline


def plot_timeseries(ax, results_mean, results_var, jobnames, jobargs,  metric, smoothed=False, first_step=10, step_size=5, 
                    error_bars=True):
    njobs = len(jobnames)
    fs = (12 if njobs <= 6 else 8)
    colors = cm.tab20(np.linspace(0, 1, njobs))
    markers = []
    labels = []
    i = 0
    for jobname in jobnames:
        res = jobargs[jobname]
        agent_name = res['agent_name']
        full_name = res['agent_full_name']
        parts = parse_full_name(full_name)
        algo, ef, lin, niter = parts['algo'], parts['ef'], parts['lin'], parts['niter']

        means = results_mean[jobname]
        std_devs = jnp.sqrt(results_var[jobname] + 1e-10)
        elapsed_mean = res['elapsed_mean']
        elapsed_std = jnp.sqrt(res['elapsed_var'] + 1e-10)

        nans = jnp.isnan(means)
        if jnp.any(nans):
            T = find_first_true(nans)
        else:
            T = len(means)
        if T<first_step:
            #print(f'skipping {jobname}/{metric}, only {T} non-nans ')
            continue       
        final_mean = means[T-1]
        final_std = std_devs[T-1]

        plot_params = make_plot_params(algo, ef, lin)
        expt_name =  f'{agent_name} [sec:{elapsed_mean:.1f}+-{elapsed_std:.1f}'
        
        if final_mean > 1000:
            final_mean_str = f'{final_mean:.2e}'
        else:
            final_mean_str = f'{final_mean:.2f}'
        final_std_str = f'{final_std:.2f}'

        expt_name = expt_name + f' {metric}@T={T}:{final_mean_str}+-{final_std_str}]'


        steps = jnp.arange(0, T)
        ndx = steps[first_step:T:step_size] #  skip first K time steps, since it messes up the vertical scale
        xs = steps[ndx]
        ys = means[ndx]
        lower_bound = means[ndx] - std_devs[ndx]
        upper_bound = means[ndx] + std_devs[ndx]
        T = len(xs)

        if not error_bars:
            line, = ax.plot(xs, ys, color=colors[i], label=expt_name) 
        else:
            ax.fill_between(xs, lower_bound, upper_bound, alpha=0.2, color=colors[i], label=expt_name) 
        labels.append(expt_name)

        Tstep = int(T/20)
        if Tstep==0:
            markers_on = np.arange(0, T)
        else:
            markers_on = np.arange(0, T, Tstep)
        jittered_x = add_jitter(xs[markers_on], jitter_amount=int(0.01*T))
        marker = plot_params['marker']
        ax.scatter(jittered_x, ys[markers_on], color=colors[i], marker=marker, s=12, label=expt_name)
        markers.append(marker)
   
        i = i + 1 # This counts actual number of lines, which may be less than njobs due to exclusioh

    num_lines = i
    ax.grid(True)
    legend_handles = [Line2D([0], [0], color=colors[i], marker=markers[i], linestyle='-', label=labels[i]) for i in range(num_lines)]
    ax.legend(handles=legend_handles, loc='upper left')
    #ax.legend(loc='upper right', prop={'size': fs})
    ax.set_ylabel(metric, fontsize=12)
    ax.set_xlabel('num. training observations', fontsize=12)
    #ax.set_ylim(stats['qlow'], stats['qhigh']) # truncate outliers
        
    return ax

def plot_and_save(results_dir, results_mean, results_var, jobargs,  metric, fig_dir, use_log=False, smoothed=False,  
            exclude='', include='', name='', first_step=10, step_size=5, error_bars=True, include_baseline=True,
            ymin='None', ymax='None'):
    jobnames = filter_jobnames(jobargs, exclude, include)

    fig, ax = plt.subplots(figsize=(8,6)) # width, height in inches
    if include_baseline and ((metric == 'nlpd-pi') or (metric == 'nlpd-mc')):
        mse, nll = compute_linreg_baseline(results_dir)
        ax.axhline(y=nll, linestyle=':', label='linreg-nll')

    ax = plot_timeseries(ax, results_mean, results_var, jobnames, jobargs,  metric,  smoothed=smoothed,  
            first_step=first_step, step_size=step_size, error_bars=error_bars)
    
    if ymin != 'None':
        ax.set_ylim(bottom=int(ymin))
    if ymax != 'None':
        ax.set_ylim(top=int(ymax))

    jobname = jobnames[0]
    model_name = jobargs[jobname]['model_name']
    data_name = jobargs[jobname]['data_name']
    ttl = f'Model: {model_name}. Data: {data_name}. Expt: {name}'
    ax.set_title(ttl, fontsize=10)

    if len(name)>0:
        fname = f"{fig_dir}/{name}_{metric}"
    else:
        fname = f"{fig_dir}/results_{metric}"
    if use_log:
        ax.set_yscale('log')
        fname = fname + "_log"
    #if smoothed:
    #    fname = fname + "_smoothed"
    if include_baseline:
        fname = fname + "_baseline"
    if error_bars:
        fname = fname + "_error_bars"
    if ymin != 'None':
        fname = fname + f"_ymin{ymin}"
    if ymax != 'None':
        fname = fname + f"_ymax{ymax}"
    if first_step>0:
        fname = fname + f"_from{first_step}"
    print(f'Saving figure to {fname}')
    fig.savefig(f'{fname}.png', bbox_inches='tight', dpi=300)
    fig.savefig(f'{fname}.pdf', bbox_inches='tight', dpi=300)


def main(args):
    results_dir = args.dir
    fig_dir = f'{results_dir}/figs'
    print(f'Writing plots to {fig_dir}')
    path = Path(fig_dir)
    path.mkdir(parents=True, exist_ok=True)   
    #make_file_with_timestamp(results_dir)
    metrics = extract_metrics_from_files(results_dir, args.jobs_file, args.jobs_suffix,
        exclude_val=True, remove_mean=True, exclude_mse=True)
    print(metrics)
    metrics = ['nlpd-pi']
    jobargs = extract_jobargs(results_dir,  args.jobs_file, args.jobs_suffix)

    for metric in metrics:
        print(metric)
        results_mean = extract_results(results_dir,  f'{metric}_mean', args.jobs_file, args.jobs_suffix)
        results_var = extract_results(results_dir,  f'{metric}_var', args.jobs_file, args.jobs_suffix)
        plot_and_save(results_dir, results_mean, results_var, jobargs, metric, fig_dir,  use_log=False,
                      exclude=args.exclude, include=args.include, error_bars=True, ymin=args.ymin, ymax=args.ymax,
                    name=args.name, first_step=args.first_step, smoothed=True, include_baseline=args.include_baseline) 

        plot_and_save(results_dir, results_mean, results_var, jobargs, metric, fig_dir,  use_log=False,
                      exclude=args.exclude, include=args.include, error_bars=False, ymin=args.ymin, ymax=args.ymax,
                    name=args.name, first_step=args.first_step, smoothed=True, include_baseline=args.include_baseline) 

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="")
    parser.add_argument("--jobs_file", type=str, default="jobs.csv")
    parser.add_argument("--jobs_suffix", type=str, default="-averaged")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--first_step", type=int, default=10)
    parser.add_argument("--exclude", type=str, default="")
    parser.add_argument("--include", type=str, default="")
    parser.add_argument("--error_bars", type=int, default=1)
    parser.add_argument("--include_baseline", type=int, default=0)
    parser.add_argument("--ymin", type=str, default='None')
    parser.add_argument("--ymax", type=str, default='None')

    args = parser.parse_args()
    print(args)
    main(args)