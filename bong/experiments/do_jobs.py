

import argparse
import os
import itertools
import pandas as pd
from pathlib import Path
import os
import datetime


def main(args):
    fname = f"{args.dir}/jobs.csv"
    df_cmds = pd.read_csv(fname)
    njobs = len(df_cmds)
    print(f'Running {njobs} jobs, please be patient')

    cmd_dict = {}
    for index, row in df_cmds.iterrows():
        cmd_dict[row.jobname] = row.command

    if args.parallel:
        from lightning_sdk import Studio, Machine
        studio = Studio()
        studio.install_plugin('jobs')
        job_plugin = studio.installed_plugins['jobs']
        #jobs_dir = '/teamspace/jobs'
        #print(f'Will store results in /teamspace/jobs/[jobname]/work')

        n = 0
        output = {}
        for jobname, cmd in cmd_dict.items():
            print(f'\n Queuing job {n} of {njobs}:\n{cmd}')
            #output_dir = f'{jobs_dir}/{jobname}/work' 
            if args.machine == 'local':
                output[jobname] = job_plugin.run(cmd, name=jobname) # run on local VM
            elif args.machine == 'A10G':
                output[jobname] = job_plugin.run(cmd, machine=Machine.A10G, name=jobname)
            elif args.machine == 'cpu':
                output[jobname] = job_plugin.run(cmd, machine=Machine.CPU, name=jobname)
            else:
                raise Exception(f'Unknown machine type {args.machine}')
            print(f'Requested name {jobname}, given name {output[jobname].name}')
            n = n + 1

    else:
        jobs_dir = args.dir
        n = 0
        for jobname, cmd in cmd_dict.items():   
            output_dir = f'{jobs_dir}/{jobname}/work'  
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)   
            cmd = cmd + f' --dir {output_dir}'
            print(f'\n Running job {n} of {njobs}:\n{cmd}')
            os.system(cmd)
            n = n + 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--parallel", type=bool, default=False)
    parser.add_argument("--machine", type=str, default="local", choices=["local", "cpu", "A10G"])

    args = parser.parse_args()
    main(args)