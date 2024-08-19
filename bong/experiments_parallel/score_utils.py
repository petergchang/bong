import numpy as np
from bong.util import parse_full_name


def get_scores_per_job(results):
    scores = {}
    for job, res in results.items():
        vals = res["vals"]
        T = res["valid_len"]
        if T < len(vals):
            scores[job] = 1e10
        else:
            eval_step = int(
                0.5 * T
            )  # pick half way through validation as the metric to optimize
            scores[job] = vals[eval_step]
    return scores


def get_scores_and_expts_per_agent(results, job_scores):
    agent_scores, agent_expts, agent_jobs = {}, {}, {}
    for job, res in results.items():
        expt = res["agent_full_name"]
        parts = parse_full_name(expt)
        # name = make_agent_name_from_parts(parts['algo'], parts['param'], parts['lin'])
        name = f'{parts['algo']}_{parts['param']}_{parts['lin']}'  # uniquify
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
    best_expt = {}
    for agent in agent_names:
        scores = np.array(agent_scores[agent])
        i = np.argmin(scores)
        expts = agent_expts[agent]
        best_expt[agent] = expts[i]
    return best_expt


def filter_results_by_best(results, best_expt_per_agent):
    best_expts = best_expt_per_agent.values()
    filtered = {}
    jobnames = results.keys()
    for i, jobname in enumerate(jobnames):
        res = results[jobname]
        expt_name = res["agent_name"]
        if expt_name in best_expts:
            filtered[jobname] = results[jobname]
    return filtered


def extract_best_results_by_val_metric(dir, metric):
    results = extract_results_from_files(dir, metric)
    metric_val = f"{metric}_val"
    results_val = extract_results_from_files(dir, metric_val)
    job_scores = get_scores_per_job(results_val)
    agent_scores, agent_expts, agent_jobs = get_scores_and_expts_per_agent(
        results, job_scores
    )
    best_expt_per_agent = get_best_expt_per_agent(agent_scores, agent_expts)
    filtered = filter_results_by_best(results, best_expt_per_agent)
    return filtered


def test_filtering():
    root_dir = "/teamspace/studios/this_studio/jobs"
    data_dir = "reg-D10-mlp_20_20_1"
    model_dir = "mlp_10_10_1"
    agent_dir = "A:Any-P:Any-Lin:1-LR:Any-IT:10-MC:10-EF:1-R:10"
    dir = f"{root_dir}/{data_dir}/{model_dir}/{agent_dir}"

    metric = "nll"
    results = extract_results_from_files(dir, metric)
    best_results = extract_best_results_by_val_metric(dir, metric)
    plot_results_from_dict(results, metric)
    plot_results_from_dict(best_results, metric)
