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

from job_utils import extract_results_from_files, extract_metrics_from_files
from bong.util import make_file_with_timestamp, parse_full_name, make_full_name


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


def make_plot_params(args):
    markers = {'bong': 'o', 'blr': 's', 'bog': 'x', 'bbb': '*'}
    marker = markers[args['algo']]
    if (args['ef']==0) & (args['lin']==0):
        linestyle = '-'
    elif (args['ef']==1) & (args['lin']==0):
        linestyle = '--'
    else:
       linestyle = '..'
    return {
            'linestyle': linestyle,
            'linewidth': 1,
            'marker': marker,
            'markersize': 6
            }



def add_jitter(x, jitter_amount=0.1):
    return x + np.random.uniform(-jitter_amount, jitter_amount, size=x.shape)


def convolve_smooth(time_series, width=5): 
    kernel = jnp.ones(width) / width
    smoothed_time_series = jnp.convolve(time_series, kernel, mode='same')
    return smoothed_time_series


def plot_results(results,  metric,  first_step=10, tuned=False, smoothed=False, step_size=5, max_len=1000):
    jobnames = results.keys()
    fig, ax = plt.subplots(figsize=(8,6)) # width, height in inches
    njobs = len(jobnames)
    fs = (12 if njobs <= 6 else 8)
    times = {}
    for i, jobname in enumerate(jobnames):
        res = results[jobname]
        vals = res['vals'].to_numpy()
        agent_name = res['agent_name']
        full_name = res['agent_full_name']
        T = len(vals)
        elapsed = 1000*round(res['elapsed']/T, 3)
        times[agent_name] = elapsed # we assume each agent only occurs once
        expt_name =  f'{agent_name} [{elapsed:.1f} ms/step]'
        model_name = res['model_name']
        data_name = res['data_name']
        parts = parse_full_name(full_name)
        plot_params = make_plot_params(parts)
        ttl = f'Data:{data_name}. Model:{model_name}. Tuned:{tuned}'

        T = res['valid_len']
        T = min(T, max_len)
        steps = jnp.arange(0, T)
        ndx = steps[first_step:T:step_size] #  skip first K time steps, since it messes up the vertical scale
        xs = steps[ndx]
        ys = vals[ndx]
        if smoothed:
            ys = convolve_smooth(ys)

        #ax.plot(xs, ys, markevery=20, label=expt_name, **plot_params)
        line, = ax.plot(xs, ys, label=expt_name) # **plot_params)
        markers_on = np.arange(0, len(xs), 10)
        jittered_x = add_jitter(xs[markers_on])
        ax.plot(jittered_x, ys[markers_on], marker=plot_params['marker'],
                 linestyle='None', markersize=12, color=line.get_color())

        ax.grid(True)
        ax.legend(loc='upper right', prop={'size': fs})
        ax.set_ylabel(metric, fontsize=12)
        ax.set_xlabel('num. training observations', fontsize=12)
        ax.set_title(ttl, fontsize=10)
        #ax.set_ylim(stats['qlow'], stats['qhigh']) # truncate outliers
        
    return fig, ax

def save_plot(results_dir, metric, use_log=False, tuned=False, smoothed=False, truncated=False):
    if tuned:
        #results = extract_best_results_by_val_metric(results_dir,  metric)
        results = None
    else:
        results = extract_results_from_files(results_dir,  metric)
    #stats = extract_stats(results)
    if truncated:
        max_len = 250
    else:
        max_len = 1000
    fig, ax = plot_results(results,  metric,  tuned=tuned, smoothed=smoothed, max_len=max_len)
    fname = f"{results_dir}/{metric}"
    if use_log:
        ax.set_yscale('log')
        fname = fname + "_log"
    if tuned:
        fname = fname + "_best"
    if smoothed:
        fname = fname + "_smoothed"
    if truncated:
        fname = fname + "_truncated"
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
    print(f'Writing plots to {results_dir}')
    make_file_with_timestamp(results_dir)
    metrics = extract_metrics_from_files(results_dir)
    for metric in metrics:
        save_plot(results_dir, metric,  use_log=False, tuned=False)
        #save_plot(results_dir, metric,  use_log=False, tuned=True)
        #save_plot(results_dir, metric,  use_log=False, tuned=True, truncated=True)
        #save_plot(results_dir, metric,  use_log=False, tuned=True, smoothed=True)

        if 0:
            try:
                save_plot(results_dir, metric, use_log=True, tuned=False)
            except Exception as e:
                print(f'Could not compute logscale plot, error {e}')
            try:
                save_plot(results_dir, metric, use_log=True, tuned=True)
            except Exception as e:
                print(f'Could not compute logscale plot, error {e}')
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="")

    args = parser.parse_args()
    main(args)