

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


def extract_metrics_from_files(dir, exclude_val=True):
    fname = f"{dir}/jobs.csv"
    df = pd.read_csv(fname)
    jobnames = df['jobname']
    jobname = jobnames[0]
    fname = f"{dir}/{jobname}/work/results.csv"
    df_res = pd.read_csv(fname)
    metrics =  df_res.columns
    if exclude_val:
        metrics = [m for m in metrics if "_val" not in m ]
    return metrics

def find_first_true(arr):
    true_indices = np.where(arr)[0]
    if true_indices.size > 0:
        first_true_index = true_indices[0]
    else:
        first_true_index = None
    return first_true_index

def extract_results_from_files(dir,  metric):
    fname = f"{dir}/jobs.csv"
    df = pd.read_csv(fname)
    jobnames = df['jobname']
    results = {}
    for jobname in jobnames:
        fname = f"{dir}/{jobname}/work/results.csv"
        df_res = pd.read_csv(fname)
        vals = df_res[metric]
        nans = np.isnan(vals)
        if np.any(nans):
            T = find_first_true(nans)
        else:
            T = len(vals)
        
        fname = f"{dir}/{jobname}/work/args.json"
        with open(fname, 'r') as json_file:
            args = json.load(json_file)
        d = {
            'metric': metric,
            'vals': vals,
            'valid_len': T,
            'agent_name': args['agent_name'], 
            'model_name': args['model_name'],
            'data_name': args['data_name'],
            }
        results[jobname] = d
    return results

def extract_stats_from_results(results, qlow=0, qhigh=90, first_step=1):
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

def plot_results_from_dict(results,  metric, qlow=0, qhigh=80, first_step=1, filtered=False):
    jobnames = results.keys()
    stats = extract_stats_from_results(results, qlow, qhigh, first_step)
    fig, ax = plt.subplots(figsize=(6,6))
    qlows, qhighs = [], []
    for i, jobname in enumerate(jobnames):
        res = results[jobname]
        vals = res['vals']
        agent_name = res['agent_name']
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

        ax.plot(steps[ndx], vals[ndx], label=agent_name, **plot_params)
        ax.grid(True)
        ax.legend(loc='upper right', prop={'size': 'x-small'})
        ax.set_ylabel(metric)
        ax.set_xlabel('num. training observations')
        ax.set_title(ttl, fontsize=8)
        ax.set_ylim(stats['qlow'], 1.1*stats['qhigh']) # truncate outliers
        
    return fig, ax

def get_scores_per_job(results):
    scores = {}
    for (job, res) in results.items():
        vals = res['vals']
        T = res['valid_len']
        if T < len(vals):
            scores[job] = 1e10
        else:
            eval_step = int(0.5*T) # pick half way through validation as the metric to optimize
            scores[job] = vals[eval_step]
    return scores

def get_scores_and_expts_per_agent(results, job_scores):
    agent_scores, agent_expts, agent_jobs = {}, {}, {}
    for (job, res) in results.items():
        expt = res['agent_name']
        parts = parse_agent_full_name(expt)
        name = make_agent_name_from_parts(parts['algo'], parts['param'], parts['lin'])
        job_score = job_scores[job]
        if name in agent_scores:
            agent_scores[name].append(job_score)
            agent_expts[name].append(expt)
            agent_jobs[name].append(job)
        else:
            agent_scores[name] = [job_score]
            agent_expts[name] = [expt]
            agent_jobs[name] = [job]
    return agent_scores, agent_expts, agent_jobs

def get_best_expt_per_agent(agent_scores, agent_expts):
    agent_names = agent_scores.keys()
    agent_best_expt = {}
    for agent in agent_names:
        scores = np.array(agent_scores[agent])
        i = np.argmin(scores)
        expts = agent_expts[agent]
        agent_best_expt[agent] = expts[i]
    return agent_best_expt

def filter_results_by_best(results, best_expt_per_agent):
    best_expts = best_expt_per_agent.values()
    filtered = {}
    jobnames = results.keys()
    for i, jobname in enumerate(jobnames):
        res = results[jobname]
        expt_name = res['agent_name']
        if expt_name in best_expts:
            filtered[jobname] = results[jobname]
    return filtered

def extract_best_results_by_val_metric(dir, metric):
    results = extract_results_from_files(dir,  metric)
    metric_val = f'{metric}_val'
    results_val = extract_results_from_files(dir,  metric_val)
    job_scores = get_scores_per_job(results_val)
    agent_scores, agent_expts, agent_jobs = get_scores_and_expts_per_agent(results, job_scores)
    best_expt_per_agent = get_best_expt_per_agent(agent_scores, agent_expts)
    filtered = filter_results_by_best(results, best_expt_per_agent)
    return filtered

def test_filtering():
    root_dir = '/teamspace/studios/this_studio/jobs'
    data_dir = 'reg-D10-mlp_20_20_1'
    model_dir = 'mlp_10_10_1'
    agent_dir = 'A:Any-P:Any-Lin:1-LR:Any-IT:10-MC:10-EF:1-R:10'
    dir = f'{root_dir}/{data_dir}/{model_dir}/{agent_dir}'

    metric = 'nll'
    results = extract_results_from_files(dir,  metric)
    best_results = extract_best_results_by_val_metric(dir, metric)
    plot_results_from_dict(results, metric)
    plot_results_from_dict(best_results, metric)




###### OLD stuff below

def plot_df(df):
    #fname = "/Users/kpmurphy/github/bong/bong/results/baz_parsed.csv"
    #df = pd.read_csv(fname)

    niters = df['I'].unique()
    niters = niters[niters != 0]

    lrs = df['LR'].unique()
    lrs = lrs[lrs != 0]

    mcs = df['M'].unique()
    mcs = mcs[mcs != 0]

    agents = df['prefix'].unique()
    agents = agents[ agents != "laplace" ]
    agents = agents[ agents != "fg-bong" ]
    agents = agents[ agents != "fg-l-bong" ]

    fs = 'x-small'
    loc = 'upper right' #'lower left'

    df2 = df[ df['prefix']=='fg-l-bong']
    kl = df2['kl'].to_numpy()
    T = len(kl)
    print(T)

    # extract subset of points for plotting to avoid cluttered markers
    #ndx = jnp.array(range(0, T, 10)) # decimation of points 
    ndx = round(jnp.linspace(0, T-1, num=min(T,30)))
    # skip first 2 time steps, since it messes up the vertical scale
    ndx = ndx[2:]

    fig, axs = plt.subplots(len(niters), len(lrs), figsize=(8, 8))
    for i, niter in enumerate(niters):
        for j, lr in enumerate(lrs):
            ax = axs[i,j]
            df2 = df[ (df['I']==niter) & (df['LR']==lr) ]
            for agent in agents:
                df3 = df2[df2['prefix']==agent]
                for mc in mcs:
                    df4 = df3[ (df3['M']==mc) ]
                    name = f'{agent}-M{mc}-I{niter}-LR{lr}'
                    #print(name)
                    steps = df4['step'].to_numpy()
                    kl = df4['kl'].to_numpy()
                    if np.any(np.isnan(kl)):
                        continue
                    else:
                        ax.plot(
                            steps[ndx], kl[ndx], label=name, 
                            marker=make_marker(agent)
                        )

            if 'fg-bong' in df['prefix'].unique():
                agent = 'fg-bong' # not indexed by I,LR
                df2 = df[ (df['prefix']==agent) ]
                for mc in mcs:
                    df3 = df2[ (df2['M']==mc) ]
                    name = f'{agent}-M{mc}'
                    steps = df3['step'].to_numpy()
                    kl = df3['kl'].to_numpy()
                    ax.plot(steps[ndx], kl[ndx], label=name, marker=make_marker(agent))
            
            if 'fg-l-bong' in df['prefix'].unique():
                agent = 'fg-l-bong' # not indexed by I,LR,M
                df2 = df[df['prefix']==agent]
                name = f'{agent}'
                steps = df2['step'].to_numpy()
                kl = df2['kl'].to_numpy()
                ax.plot(steps[ndx], kl[ndx], label=name, marker=make_marker(agent))
            
            if 'laplace' in df['prefix'].unique():
                df2 = df[ (df['prefix']=='laplace') ]
                kldiv = df2['kl'].to_numpy()[0]
                ax.axhline(kldiv, color="black", linestyle="--", label='laplace')

            ax.legend(loc=loc, prop={'size': fs})
            ax.set_title(f'Iter{niter}-LR{lr}')
    
def plot_results_from_dict_old(result_dict, curr_path=None, file_prefix='', ttl='', filetype='png'):
    result_dict = result_dict.copy()
    #T = extract_nsteps_from_result_dict(result_dict)
    T = result_dict.pop('ntest')
    # extract subset of points for plotting to avoid cluttered markers
    #ndx = jnp.array(range(0, T, 10)) # decimation of points 
    ndx = round(jnp.linspace(0, T-1, num=min(T,50)))
    # skip first 2 time steps, since it messes up the vertical scale
    ndx = ndx[2:]

    fs = 'small'
    loc = 'lower left'

    # Save KL-divergence, linear scale
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for agent_name, (_, kldiv, _, _) in result_dict.items():
        if jnp.any(jnp.isnan(kldiv)):
            continue
        if agent_name == "laplace":
            ax.axhline(kldiv, color="black", linestyle="--", label=agent_name)
        else:
            ax.plot(ndx, kldiv[ndx], label=agent_name, marker=make_marker(agent_name))
    ax.set_xlabel("number of iteration")
    ax.set_ylabel("KL-divergence")
    #ax.set_yscale("log")
    ax.grid()
    #ax.legend()
    ax.legend(loc=loc, prop={'size': fs})
    ax.set_title(ttl)
    if curr_path:
        #fname = Path(curr_path, f"{file_prefix}_kl_divergence.pdf")
        #fig.savefig(fname, bbox_inches='tight', dpi=300)
        fname = Path(curr_path, f"{file_prefix}_kl_divergence.{filetype}")
        fig.savefig(fname, bbox_inches='tight', dpi=300)

    # Save KL-divergence, log scale
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for agent_name, (_, kldiv, _, _) in result_dict.items():
        if jnp.any(jnp.isnan(kldiv)):
            continue
        if agent_name == "laplace":
            ax.axhline(kldiv, color="black", linestyle="--", label=agent_name)
        else:
            ax.plot(ndx, kldiv[ndx], label=agent_name, marker=make_marker(agent_name))
    ax.set_xlabel("number of iteration")
    ax.set_ylabel("KL-divergence")
    ax.set_yscale("log")
    ax.grid()
    ax.legend(loc=loc, prop={'size': fs})
    ax.set_title(ttl)
    if curr_path:
        fname = Path(curr_path, f"{file_prefix}_kl_divergence_logscale.{filetype}")
        fig.savefig(fname, bbox_inches='tight', dpi=300)
    
    # Save NLL
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for agent_name, (_, _, nll, _) in result_dict.items():
        if jnp.any(jnp.isnan(nll)):
            continue
        if agent_name == "laplace":
            ax.axhline(nll, color="black", linestyle="--", label=agent_name)
        else:
            ax.plot(ndx, nll[ndx], label=agent_name, marker=make_marker(agent_name))
    ax.set_xlabel("number of iteration")
    ax.set_ylabel("NLL (plugin)")
    ax.grid()
    ax.legend(loc=loc, prop={'size': fs})
    ax.set_title(ttl)
    if curr_path:
        fname = Path(curr_path, f"{file_prefix}_plugin_nll.{filetype}")
        fig.savefig(fname, bbox_inches='tight', dpi=300)

      # Save NLL
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for agent_name, (_, _, nll, _) in result_dict.items():
        if jnp.any(jnp.isnan(nll)):
            continue
        if agent_name == "laplace":
            ax.axhline(nll, color="black", linestyle="--", label=agent_name)
        else:
            ax.plot(ndx, nll[ndx], label=agent_name, marker=make_marker(agent_name))
    ax.set_xlabel("number of iteration")
    ax.set_ylabel("NLL (plugin)")
    ax.set_yscale("log")
    ax.grid()
    ax.legend(loc=loc, prop={'size': fs})
    ax.set_title(ttl)
    if curr_path:
        fname = Path(curr_path, f"{file_prefix}_plugin_nll_logscale.{filetype}")
        fig.savefig(fname, bbox_inches='tight', dpi=300)
    
    # Save NLPD
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for agent_name, (_, _, _, nlpd) in result_dict.items():
        if jnp.any(jnp.isnan(nlpd)):
            continue
        if agent_name == "laplace":
            ax.axhline(nlpd, color="black", linestyle="--", label=agent_name)
        else:
            ax.plot(ndx, nlpd[ndx], label=agent_name, marker=make_marker(agent_name))
    ax.set_xlabel("number of iteration")
    ax.set_ylabel("NLPD (MC)")
    ax.grid()
    ax.legend(loc=loc, prop={'size': fs})
    ax.set_title(ttl)
    if curr_path:
        fname = Path(curr_path, f"{file_prefix}_mc_nlpd.{filetype}")
        fig.savefig(fname, bbox_inches='tight', dpi=300)

    # Save runtime
    fig, ax = plt.subplots()
    for agent_name, (runtime, _, _, _) in result_dict.items():
        ax.bar(agent_name, runtime)
    ax.set_ylabel("runtime (s)")
    plt.setp(ax.get_xticklabels(), rotation=30)
    if curr_path:
        fname = Path(curr_path, f"{file_prefix}_runtime.{filetype}")
        fig.savefig(fname, bbox_inches='tight', dpi=300)
    #plt.close('all')