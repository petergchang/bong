

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



def plot_results_from_dict(results,  metric, stats, first_step=3, filtered=False):
    jobnames = results.keys()
    fig, ax = plt.subplots(figsize=(8,6)) # width, height in inches
    for i, jobname in enumerate(jobnames):
        res = results[jobname]
        vals = res['vals']
        agent_name = res['agent_name']
        T = len(vals)
        elapsed = 1000*round(res['elapsed']/T, 3)
        expt_name =  f'{agent_name} [{elapsed} ms/step]'
        model_name = res['model_name']
        data_name = res['data_name']
        parts = parse_agent_full_name(agent_name)
        plot_params = make_plot_params(parts)
        if filtered:
            prefix = 'best'
        else:
            prefix = ''
        ttl = f'{prefix} {metric}. D={data_name}. M={model_name}'

        T = res['valid_len']
        steps = jnp.arange(0, T)
        ndx = round(jnp.linspace(0, T-1, num=min(T,30))) #    # extract subset of points for plotting to avoid cluttered markers
        ndx = ndx[first_step:] #  skip first 2 time steps, since it messes up the vertical scale

        ax.plot(steps[ndx], vals[ndx], label=expt_name, **plot_params)
        ax.grid(True)
        ax.legend(loc='upper right', prop={'size': 12})
        ax.set_ylabel(metric, fontsize=12)
        ax.set_xlabel('num. training observations', fontsize=12)
        ax.set_title(ttl, fontsize=10)
        #ax.set_ylim(stats['qlow'], stats['qhigh']) # truncate outliers
        
    return fig, ax







