import argparse
import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import json

from bong.util import parse_full_name
from bong.util import add_jitter, make_plot_params


def main(args):
    results_dir = args.dir

    fig_dir = f"{results_dir}/figs"
    print(f"Writing plots to {fig_dir}")
    path = Path(fig_dir)
    path.mkdir(parents=True, exist_ok=True)

    fname = f"{results_dir}/jobs.csv"
    df = pd.read_csv(fname)
    jobnames = df["jobname"]
    pt_dict_per_agent = {}  # (param, time) pair per agent
    full_name_dict = {}
    for i, jobname in enumerate(jobnames):
        fname = f"{results_dir}/jobs/{jobname}/args.json"
        if not os.path.isfile(fname):
            print("This file does not exist, skipping:", fname)
            continue
        with open(fname, "r") as json_file:
            res = json.load(json_file)

        full_name = res["agent_full_name"]
        agent_name = res["agent_name"]
        full_name_dict[agent_name] = full_name  # so we can later retrieve attributes

        nparams = res["model_nparams"]
        T = res["ntrain"]
        elapsed = res["elapsed"]
        steptime = elapsed / T
        if agent_name in pt_dict_per_agent:
            times_per_param = pt_dict_per_agent[agent_name]
            times_per_param[nparams] = steptime
            pt_dict_per_agent[agent_name] = times_per_param
        else:
            times_per_param = {nparams: steptime}
            pt_dict_per_agent[agent_name] = times_per_param

    agent_names = full_name_dict.keys()
    nparams_per_agent = {}
    times_per_agent = {}
    for agent in agent_names:
        times_per_param = pt_dict_per_agent[agent]
        nparams_per_agent[agent] = np.array(list(times_per_param.keys()))
        times_per_agent[agent] = np.array(list(times_per_param.values()))

    fig, ax = make_figure(nparams_per_agent, times_per_agent, full_name_dict)
    fname = f"{fig_dir}/times"
    print(f"Saving figure to {fname}")
    fig.savefig(f"{fname}.png", bbox_inches="tight", dpi=300)
    fig.savefig(f"{fname}.pdf", bbox_inches="tight", dpi=300)

    fig, ax = make_figure(
        nparams_per_agent, times_per_agent, full_name_dict, jitter=True
    )
    # ax.set_yscale('log')
    fname = f"{fig_dir}/times_jitter"
    print(f"Saving figure to {fname}")
    fig.savefig(f"{fname}.png", bbox_inches="tight", dpi=300)
    fig.savefig(f"{fname}.pdf", bbox_inches="tight", dpi=300)


def make_figure(nparams_per_agent, times_per_agent, full_name_per_agent, jitter=False):
    agent_names = full_name_per_agent.keys()
    fig, ax = plt.subplots(figsize=(8, 6))  # width, height in inches
    for agent in agent_names:
        agent_name_long = full_name_per_agent[agent]
        parts = parse_full_name(agent_name_long)
        algo, param, lin, ef = parts["algo"], parts["param"], parts["lin"], parts["ef"]
        kwargs = make_plot_params(algo, ef, lin)
        xs, ys = nparams_per_agent[agent], times_per_agent[agent]
        if jitter:
            xs = add_jitter(xs, jitter_amount=10)
        ax.plot(xs, ys, label=agent, **kwargs)
    ax.grid()
    ax.legend()
    ax.set_ylabel("Elapsed time per step (sec)")
    ax.set_xlabel("Num. parameters")
    return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="")

    args = parser.parse_args()
    main(args)
