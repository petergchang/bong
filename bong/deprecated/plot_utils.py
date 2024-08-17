

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from pathlib import Path
import os
import matplotlib.image as mpimg
import json
from bong.agents import parse_agent_full_name, make_agent_name_from_parts

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

marker_types = {
    'point': '.',
    'pixel': ',',
    'circle': 'o',
    'triangle_down': 'v',
    'triangle_up': '^',
    'triangle_left': '<',
    'triangle_right': '>',
    'tri_down': '1',
    'tri_up': '2',
    'tri_left': '3',
    'tri_right': '4',
    'square': 's',
    'pentagon': 'p',
    'star': '*',
    'hexagon1': 'h',
    'hexagon2': 'H',
    'plus': '+',
    'x': 'x',
    'diamond': 'D',
    'thin_diamond': 'd',
    'vline': '|',
    'hline': '_'
}




def plot_results_from_images(root_dir, data_dir, model_dir, agent_dir, metrics=['nll',  'nlpd']):
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





def make_plot_params(args):
    markers = {'bong': 'o', 'blr': 's', 'bog': 'x', 'bbb': '*'}
    marker = markers[args['algo']]
    if (args['ef']==0) & (args['lin']==0): linestyle = '-'
    if (args['ef']==1) & (args['lin']==0): linestyle = '--'
    if (args['ef']==0) & (args['lin']==1): linestyle = ':'
    if (args['ef']==1) & (args['lin']==1): linestyle = '-.'
    return {
            'linestyle': linestyle,
            'linewidth': 1,
            'marker': marker,
            'markersize': 6
            }


def convolve_smooth(time_series, width=5): 
    kernel = jnp.ones(width) / width
    smoothed_time_series = jnp.convolve(time_series, kernel, mode='same')
    return smoothed_time_series

def add_jitter(x, jitter_amount=0.1):
    return x + np.random.uniform(-jitter_amount, jitter_amount, size=x.shape)


def plot_results(results,  metric, stats, first_step=10, tuned=False, smoothed=False, step_size=5, max_len=1000):
    jobnames = results.keys()
    fig, ax = plt.subplots(figsize=(8,6)) # width, height in inches
    njobs = len(jobnames)
    fs = (12 if njobs <= 6 else 8)
    times = {}
    for i, jobname in enumerate(jobnames):
        res = results[jobname]
        vals = res['vals'].to_numpy()
        agent_name = res['agent_name']
        T = len(vals)
        elapsed = 1000*round(res['elapsed']/T, 3)
        times[agent_name] = elapsed # we assume each agent only occurs once
        expt_name =  f'{agent_name} [{elapsed:.1f} ms/step]'
        model_name = res['model_name']
        data_name = res['data_name']
        parts = parse_agent_full_name(agent_name)
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
                 linestyle='None', markersize=8, color=line.get_color())

        ax.grid(True)
        ax.legend(loc='upper right', prop={'size': fs})
        ax.set_ylabel(metric, fontsize=12)
        ax.set_xlabel('num. training observations', fontsize=12)
        ax.set_title(ttl, fontsize=10)
        #ax.set_ylim(stats['qlow'], stats['qhigh']) # truncate outliers
        
    return fig, ax


def plot_times(results):
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




