

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from pathlib import Path

def make_marker(name):
    #https://matplotlib.org/stable/api/markers_api.html
    markers = {'bong': 'o', 'blr': 's', 'bog': 'x', 'bbb': '*'}
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
    
#'/teamspace/studios/this_studio/jobs/{jobname}/work'
#f'/teamspace/jobs/{jobname}/work'


def plot_results_from_files(dir,  metric, save_fig=False):
    fname = f"{dir}/flags.csv"
    df_flags = pd.read_csv(fname)
    jobnames = df_flags['jobname']

    fig, ax = plt.subplots(figsize=(5,5))
    fs = 'x-small'
    loc = 'upper right' #'lower left'

    for jobname in jobnames:
        fname = f"{dir}/{jobname}/work/results.csv"
        df_res = pd.read_csv(fname)
        vals = df_res[metric]
        if np.any(np.isnan(vals)):
            print(f'found NaNs in {metric} in {fname}, skipping')
            continue
        
        agent_name = df_res['agent_name'][0] # bong_fc-MC10
        agent_type = agent_name.split('_')[0]
        #print(f'plotting {agent_name}, {agent_type}')
        marker = make_marker(agent_type)
        data_name = df_res['dataset_name'][0]
        ttl = f'{metric} on {data_name}'

        T = len(vals)
        steps = jnp.arange(0, T)
        ndx = round(jnp.linspace(0, T-1, num=min(T,30))) #    # extract subset of points for plotting to avoid cluttered markers
        ndx = ndx[2:] #  skip first 2 time steps, since it messes up the vertical scale

        ax.plot(steps[ndx], vals[ndx], label=agent_name, marker=marker)
        ax.grid(True)
        ax.legend(loc=loc, prop={'size': fs})
        ax.set_ylabel(metric)
        ax.set_xlabel('num. training observations')
        ax.set_title(ttl)
    if save_fig:
        fname = f"{dir}/{metric}.png"
        print(f'Saving figure to {fname}')
        fig.savefig(fname, bbox_inches='tight', dpi=300)

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
    
def plot_results_from_dict(result_dict, curr_path=None, file_prefix='', ttl='', filetype='png'):
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