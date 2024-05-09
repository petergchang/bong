

from lightning_sdk import Studio, Machine
import argparse
import os
import itertools
import pandas as pd
from pathlib import Path

from bong.agents import AGENT_DICT, AGENT_NAMES

def make_cmd(agent, lr, niter, nsample):
    main_name = '/teamspace/studios/this_studio/bong/bong/experiments/main.py'
    cmd = f'python {main_name} --agent {agent} --lr {lr} --niter {niter} --nsample {nsample}'
    return cmd


def extract_args(props, learning_rate, num_iter, num_sample):
    args = props.copy()
    if props['lr'] is None: args['lr'] = learning_rate
    if props['niter'] is None: args['niter'] = int(num_iter)
    if props['nsample'] is None: args['nsample'] = int(num_sample)
    args['ef'] = props['ef'] 
    args['linplugin'] = props['linplugin']
    return args

def make_results_dirname(jobname, parallel):
    # the results artefacts are written (by lightning ai studio) to a certain directory
    # depending on jobname.
    if parallel:
        output_dir = f'/teamspace/jobs/{jobname}/work'
    else:
        output_dir = f'/teamspace/studios/this_studio/jobs/{jobname}/work'
    return output_dir

def make_df_for_arg_crossproduct(agents, lrs, niters, nsamples, parallel):
    args_list = []
    for agent in agents:
        props = AGENT_DICT[agent]
        for lr in lrs:
            for niter in niters:
                for nsample in nsamples:
                    args = extract_args(props, lr, niter, nsample)
                    args['agent'] = agent
                    args_list.append(args)
    df = pd.DataFrame(args_list)
    df = df.drop_duplicates()
    N = len(df)
    jobnames = [f'job-{i}' for i in range(N)] 
    df['jobname'] = jobnames
    dirs = [make_results_dirname(j, parallel) for j in jobnames]
    df['results_dir'] = dirs
    return df

def old(args):
    grid_search_params = list(itertools.product(args.agent, args.learning_rate, args.num_iter))
    #grid_search_params = [(lr, agent) for lr in args.learning_rate for agent in args.agent]
    #for index, (agent, lr, niter) in enumerate(grid_search_params):
    #    cmd = make_cmd(agent, lr, niter)
    #    job_name = f'bong-{index}'
    #    output_dir = f'/teamspace/studios/this_studio/jobs/{job_name}/work'
    #    cmd = cmd + f' --dir {output_dir}'
    #    print('running', cmd)
    #    os.system(cmd)


def main(args):
    df = make_df_for_arg_crossproduct(args.agents, args.lrs, args.niters, args.nsamples, args.parallel)
    df['dataset'] = args.dataset
    fname = Path(args.dir, "jobs.csv")
    print("Saving to", fname)
    df.to_csv(fname, index=False) 

    cmd_dict = {}
    for index, row in df.iterrows():
        cmd = make_cmd(row.agent, row.lr, row.niter, row.nsample)
        cmd_dict[row.jobname] = cmd

    if args.parallel:
        print('parallel')
        studio = Studio()
        studio.install_plugin('jobs')
        job_plugin = studio.installed_plugins['jobs']

        for jobname, cmd in cmd_dict.items():
            output_dir = make_results_dirname(jobname, args.parallel)
            #cmd = cmd + f' --dir {output_dir}' # not needed
            print('queuing job', jobname, 'to run', cmd)
            job_plugin.run(cmd, name=jobname)
    else:
        print('serial')
        for jobname, cmd in cmd_dict.items():   
            output_dir = make_results_dirname(jobname, args.parallel)         
            cmd = cmd + f' --dir {output_dir}'
            print('running', cmd)
            os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="", help="directory in which to store jobs.csv") 
    parser.add_argument("--parallel", type=bool, default=False)

    # Data parameters
    parser.add_argument("--dataset", type=str, default="linreg")
    parser.add_argument("--key", type=int, default=0)
    parser.add_argument("--ntrain", type=int, default=500)
    parser.add_argument("--nval", type=int, default=500)
    parser.add_argument("--ntest", type=int, default=500)
    parser.add_argument("--data_dim", type=int, default=10)
    parser.add_argument("--emission_noise", type=float, default=1.0)
    

    parser.add_argument("--agents", type=str, nargs="+",
                        default=["bong-fc", "blr-fc"], choices=AGENT_NAMES)
    parser.add_argument("--lrs", type=float, nargs="+", 
                    default=[0.01, 0.05])
    parser.add_argument("--niters", type=int, nargs="+", 
                    default=[10,100])
    parser.add_argument("--nsamples", type=int, nargs="+", 
                    default=[10])

    args = parser.parse_args()
    
    main(args)

'''
python run_jobs.py  --lrs 0.001 0.01 --agents bong-fc blr-fc
'''