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
import matplotlib.cm as cm

from job_utils import extract_results_from_files, extract_metrics_from_files, append_results_with_baselines
from bong.util import make_file_with_timestamp, parse_full_name, make_full_name
from bong.util import add_jitter, convolve_smooth, make_plot_params


def plot_results_from_images_old(root_dir, data_dir, model_dir, agent_dir, metrics=['nll',  'nlpd']):
    results_dir = f"{root_dir}/{data_dir}/{model_dir}/{agent_dir}"
    ncols = len(metrics)
    fig, axs = plt.subplots(1, ncols, figsize=(16, 8))
    for i, metric in enumerate(metrics):
        fname = f'{results_dir}/{metric}.png'
        img = mpimg.imread(fname)
        ax = axs[i]
        ax.imshow(img)
        ax.axis('off')
    ttl = f'{data_dir}/{agent_dir}'
    y_offset = {1: 1, 2: 0.9, 3: 0.8}
    fig.suptitle(ttl, y=y_offset[ncols])

# keep for backwards comptatility
def make_marker(name):
    #https://matplotlib.org/stable/api/markers_api.html
    markers = {'bong': 'o', 'blr': 's', 'bog': 'x', 'bbb': '+'}
    name = name.lower()
    if "bong" in name:
        return markers['bong']
    elif "blr" in name:
        return markers['blr']
    elif "bog" in name:
        return markers['bog']
    elif "bbb" in name:
        return markers['bbb']
    else:
        return '.'



def plot_timeseries(results,  metric, smoothed=False, first_step=10, step_size=5, max_len=100_000,
                exclude='', include=''):
    jobnames = results.keys()
    fig, ax = plt.subplots(figsize=(8,6)) # width, height in inches
    njobs = len(jobnames)
    fs = (12 if njobs <= 6 else 8)
    njobs = len(jobnames)
    colors = cm.tab20(np.linspace(0, 1, njobs))
    for i, jobname in enumerate(jobnames):
        res = results[jobname]
        vals = res['vals']
        agent_name = res['agent_name']
        full_name = res['agent_full_name']
        parts = parse_full_name(full_name)
        algo, ef, lin, niter = parts['algo'], parts['ef'], parts['lin'], parts['niter']
        T = len(vals)
        #elapsed = 1000*round(res['elapsed']/T) # milliseconds per step
        elapsed = res['elapsed'] #/T # iseconds per step

        
        if full_name == 'baseline':
            plot_params = {'linestyle': ':', 'linewidth': 2}
            expt_name =  f'{agent_name}'
        else:
            plot_params = make_plot_params(algo, ef, lin)
            expt_name =  f'{agent_name} [{elapsed:.1f} s]'
        
        if len(exclude) > 1:
            skip_result = eval(exclude)
            if skip_result: continue
        if len(include) > 1:
            keep_result = eval(include)
            if not keep_result: continue


        T = res['valid_len']
        T = min(T, max_len)
        steps = jnp.arange(0, T)
        ndx = steps[first_step:T:step_size] #  skip first K time steps, since it messes up the vertical scale
        xs = steps[ndx]
        ys = vals[ndx]
        if smoothed and len(vals)==res['valid_len']:
            ys = convolve_smooth(ys, width=5, mode='valid')
            # eed to truncate xs for valid kernel
            xs = xs[2:]
            xs = xs[:-2]

        ax.plot(xs, ys, markevery=20, label=expt_name, color=colors[i], **plot_params)
        
        if 0:
            line, = ax.plot(xs, ys) #, label=expt_name) # **plot_params)
            markers_on = np.arange(0, len(xs), 10)
            jittered_x = add_jitter(xs[markers_on])
            ax.plot(jittered_x, ys[markers_on], color=line.get_color(), **plot_params, label=expt_name)

        ax.grid(True)
        ax.legend(loc='upper right', prop={'size': fs})
        ax.set_ylabel(metric, fontsize=12)
        ax.set_xlabel('num. training observations', fontsize=12)
        #ax.set_ylim(stats['qlow'], stats['qhigh']) # truncate outliers
        
    return fig, ax

def plot_and_save(results,  metric, fig_dir, use_log=False, smoothed=False, truncated=False, 
            exclude='', include='', name='', first_step=5):
    jobnames = list(results.keys())
    jobname = jobnames[0]
    res = results[jobname] # we assume all jobs use the same model and data
    model_name = res['model_name']
    data_name = res['data_name']

    #stats = extract_stats(results)
    max_len = (250 if truncated else 100_000)
    fig, ax = plot_timeseries(results,  metric,  smoothed=smoothed, max_len=max_len, 
            exclude=exclude, include=include, first_step=first_step)
    ttl = f'{name} Model {model_name}. Data {data_name}'
    ax.set_title(ttl, fontsize=10)

    fname = f"{fig_dir}/{metric}"
    if use_log:
        ax.set_yscale('log')
        fname = fname + "_log"
    if smoothed:
        fname = fname + "_smoothed"
    if truncated:
        fname = fname + "_truncated"
    if len(name)>0:
        fname = fname + f"_{name}"
    print(f'Saving figure to {fname}')
    fig.savefig(f'{fname}.png', bbox_inches='tight', dpi=300)
    fig.savefig(f'{fname}.pdf', bbox_inches='tight', dpi=300)

def extract_stats(results, qlow=0, qhigh=90, first_step=1):
    infty = 1e10
    qlows, qhighs, mins, maxs = [infty], [0], [infty], [0]
    for i, job in enumerate(results.keys()):
        res = results[job]
        vals = res['vals']
        agent_name = res['agent_name']
        T = res['valid_len']
        vals = vals[first_step:T]
        if len(vals) > 0:
            q_low, q_high = np.percentile(vals, [qlow, qhigh])
            qlows.append(q_low)
            qhighs.append(q_high)
            mins.append(np.min(vals))
            maxs.append(np.max(vals))
    d = {
        'min': np.min(np.array(mins)),
        'max': np.max(np.array(maxs)),
        'qlow': np.min(np.array(qlows)),
        'qhigh': np.max(np.array(qhighs))
    }
    return d

def main(args):
    results_dir = args.dir
    fig_dir = f'{results_dir}/figs'
    print(f'Writing plots to {fig_dir}')
    path = Path(fig_dir)
    path.mkdir(parents=True, exist_ok=True)   
    #make_file_with_timestamp(results_dir)
    metrics = extract_metrics_from_files(results_dir,  jobs_file=args.jobs_file, exclude_val=True)
    for metric in metrics:
        results = extract_results_from_files(results_dir,  metric, args.jobs_file)
        results = append_results_with_baselines(results, results_dir,  metric, args.jobs_file)
        plot_and_save(results, metric, fig_dir,  use_log=False,  exclude=args.exclude, include=args.include,
                    name=args.name, first_step=args.first_step)    
        plot_and_save(results, metric, fig_dir,  use_log=False,  exclude=args.exclude, include=args.include,
                    name=args.name, first_step=args.first_step, smoothed=True) 
        #plot_and_save(results, metric, fig_dir,  use_log=True,  exclude=args.exclude, include=args.include,
        #            name=args.name, first_step=args.first_step, smoothed=True)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="")
    parser.add_argument("--jobs_file", type=str, default="jobs.csv")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--first_step", type=int, default=5)
    parser.add_argument("--exclude", type=str, default="")
    parser.add_argument("--include", type=str, default="")

    args = parser.parse_args()
    main(args)