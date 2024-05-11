

import argparse
import os
import itertools
import pandas as pd
from pathlib import Path

from bong.agents import AGENT_DICT, AGENT_NAMES

import os
cwd = Path(os.getcwd())
root = cwd
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

def make_results_dirname_old(jobname, parallel):
    # the results artefacts are written (by lightning ai studio) to a certain directory
    # depending on jobname.
    if parallel:
        output_dir = f'/teamspace/jobs/{jobname}/work'
    else:
        output_dir = f'/teamspace/studios/this_studio/jobs/{jobname}/work'
    return output_dir



def extract_optional_agent_args(props, learning_rate, num_iter, num_sample, ef, rank):
    # Givem all the possible flag values, extract the ones needed for this agent.
    # This prevents us creating multiple agents with irrelevant arguments that differ,
    # which would cause us to create unnecessary jobs.
    args = props.copy()
    if props['lr'] is None: args['lr'] = learning_rate
    if props['niter'] is None: args['niter'] = int(num_iter)
    if props['nsample'] is None: args['nsample'] = int(num_sample)
    if props['dlr_rank'] is None: args['dlr_rank'] = int(rank)
    if props['ef'] is None: args['ef']= int(ef) 
    args['linplugin'] = props['linplugin'] # derived from agent name, not a flag
    del args['constructor']
    return args


def make_unix_cmd_given_flags(agent, lr, niter, nsample, linplugin, ef, model_neurons, rank,
                            dataset, data_dim, data_key):
    # We must pass in all flags where we want to override the default in run_job
    #main_name = '/teamspace/studios/this_studio/bong/bong/experiments/run_job.py'
    main_name = f'{script_dir}/run_job.py'
    cmd = (
        f'python {main_name} --agent {agent}  --lr {lr}'
        f' --niter {niter} --nsample {nsample} --linplugin {linplugin}'
        f' --ef {ef} --model_neurons {model_neurons} --rank {rank}'
        f' --dataset {dataset} --data_dim {data_dim} --data_key {data_key}'
    )
    return cmd


def make_df_for_flag_crossproduct(
        agent_list, lr_list, niter_list, nsample_list, ef_list, model_neurons_list, rank_list,
        data_dim_list, data_key_list):
    args_list = []
    for agent in agent_list:
        props = AGENT_DICT[agent]
        for lr in lr_list:
            for niter in niter_list:
                for nsample in nsample_list:
                    for ef in ef_list:
                        for model_neurons in model_neurons_list:
                            for rank in rank_list:
                                for data_dim in data_dim_list:
                                    for data_key in data_key_list:
                                        args = extract_optional_agent_args(props, lr, niter, nsample, ef, rank)
                                        args['agent'] = agent
                                        args['model_neurons'] = model_neurons
                                        args['data_dim'] = data_dim
                                        args['data_key'] = data_key
                                        args_list.append(args)
    df = pd.DataFrame(args_list)
    df = df.drop_duplicates()
    N = len(df)
    jobnames = [f'job-{i}' for i in range(N)] 
    df['jobname'] = jobnames
    #dirs = [make_results_dirname(j, parallel) for j in jobnames]
    #df['results_dir'] = dirs
    return df



def main(args):
    df_flags = make_df_for_flag_crossproduct(
        args.agent_list, args.lr_list, args.niter_list, args.nsample_list,
        args.ef_list, args.model_neurons_list, args.rank_list,
        args.data_dim_list, args.data_key_list)
    # for flags that are shared across all jobs, we create extra columns (duplicated across rows)
    df_flags['dataset'] = args.dataset

    cmd_dict = {}
    for index, row in df_flags.iterrows():
        cmd = make_unix_cmd_given_flags(
            row.agent, row.lr, row.niter, row.nsample,
            row.linplugin, row.ef, row.model_neurons, row.dlr_rank, # rank is a reserved word in pandas
            row.dataset, row.data_dim, row.data_key)
        cmd_dict[row.jobname] = cmd

    # Store csv containing all the flags/commands that are being executed
    path = Path(args.dir)
    path.mkdir(parents=True, exist_ok=True)
    fname = Path(path, "flags.csv")
    print("Saving to", fname)
    df_flags.to_csv(fname, index=False) 

    cmds = [{'agent': key, 'command': value} for key, value in cmd_dict.items()]
    df_cmds = pd.DataFrame(cmds)
    fname = Path(path, "cmds.csv")
    print("Saving to", fname)
    df_cmds.to_csv(fname, index=False)


    if args.parallel:
        from lightning_sdk import Studio, Machine
        studio = Studio()
        studio.install_plugin('jobs')
        job_plugin = studio.installed_plugins['jobs']
        print(f'Will store results in /teamspace/jobs/[jobname]/work')

        for jobname, cmd in cmd_dict.items():
            print('queuing job', jobname)
            print(cmd)
            job_plugin.run(cmd, name=jobname)

    else:
        #print(f'Storing results in {args.dir}')
        for jobname, cmd in cmd_dict.items():   
            output_dir = f'{args.dir}/{jobname}'  
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)   
            cmd = cmd + f' --dir {output_dir}'
            print('\n\nRunning', cmd)
            print(f'Storing results in {output_dir}')
            os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="", help="directory in which to store results") 
    parser.add_argument("--parallel", type=bool, default=False)

    # Data parameters
    parser.add_argument("--dataset", type=str, default="linreg")
    parser.add_argument("--data_dim_list", type=int, nargs="+", default=[10])
    parser.add_argument("--data_key_list", type=int, nargs="+", default=[0])
    
    # Agent parameters
    parser.add_argument("--agent_list", type=str, nargs="+", default=["bong-fc"])  # choices=AGENT_NAMES
    parser.add_argument("--lr_list", type=float, nargs="+", default=[0.01])
    parser.add_argument("--niter_list", type=int, nargs="+", default=[10])
    parser.add_argument("--nsample_list", type=int, nargs="+", default=[10])
    parser.add_argument("--ef_list", type=int, nargs="+", default=[1])
    parser.add_argument("--rank_list", type=int, nargs="+", default=[10])
    parser.add_argument("--model_neurons_list", type=str, nargs="+",  default=["1"]) # 10-10-1


    args = parser.parse_args()
    
    main(args)

'''

python run_jobs.py   --agent_list bbb-fc  --lr_list 0.01  \
    --nsample_list 10 --niter_list 10 --ef_list 0  \
    --dataset linreg --data_dim_list 10 --dir ~/jobs/linreg10

python run_jobs.py   --agent_list bong-fc blr-fc bog-fc bbb-fc --lr_list 0.005 0.01 0.05 \
    --nsample_list 10 --niter_list 10 --ef_list 0  \
    --dataset linreg --data_dim_list 10 --dir ~/jobs/linreg10
'''