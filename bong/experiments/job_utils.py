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


cwd = Path(os.getcwd())
root = cwd
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

def make_unix_cmd_given_flags(agent, lr, niter, nsample, linplugin, ef, rank,
                            model_type, model_neurons_str,
                            dataset, data_dim, dgp_type, dgp_neurons_str, ntrain):
    # We must pass in all flags where we want to override the default in run_job
    #main_name = '/teamspace/studios/this_studio/bong/bong/experiments/run_job.py'
    main_name = f'{script_dir}/run_job.py'
    #model_neurons = unmake_neuron_str(model_neurons_str)
    #dgp_neurons = unmake_neuron_str(dgp_neurons_str)
    cmd = (
        f'python {main_name} --agent {agent}  --lr {lr}'
        f' --niter {niter} --nsample {nsample} --lin {linplugin}'
        f' --ef {ef} --rank {rank}'
        f' --model_type {model_type} --model_neurons_str {model_neurons_str}'
        f' --dataset {dataset} --data_dim {data_dim}'
        f' --dgp_type {dgp_type} --dgp_neurons_str {dgp_neurons_str}'
        f' --ntrain {ntrain}'
    )
    return cmd

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
        agent_name = args['agent_name']
        d = {
            'metric': metric,
            'vals': vals,
            'valid_len': T,
            'agent_name': agent_name,
            'model_name': args['model_name'],
            'data_name': args['data_name'],
            'elapsed': args['elapsed'],
            }
        results[jobname] = d
    return results



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