

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

    # As a precaution, we make a copy of jobs.csv 
    src = f'{args.dir}/jobs.csv'
    dst = f'{args.dir}/jobs-original.csv'
    cmd = f'cp {src} {dst}' # keep copy of original
    print(cmd)
    os.system(cmd)

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
        real_jobname = {}
        output_per_job = {}
        for jobname, cmd in cmd_dict.items():
            print(f'\n Queuing job {n} of {njobs}:\n{cmd}')
            #output_dir = f'{jobs_dir}/{jobname}/work' 
            if args.machine == 'local':
                result = job_plugin.run(cmd, name=jobname) # run on local VM
            elif args.machine == 'A10G':
                result = job_plugin.run(cmd, machine=Machine.A10G, name=jobname)
            elif args.machine == 'cpu':
                result = job_plugin.run(cmd, machine=Machine.CPU, name=jobname)
            else:
                raise Exception(f'Unknown machine type {args.machine}')
            output_per_job[jobname] = result
            print(result)
            real_jobname[jobname] = output_per_job[jobname].name
            n = n + 1
        print('Actual names of launched jobs')
        print(real_jobname)

    else:
        import subprocess
        output_per_job = {}

        n = 0
        for jobname, cmd in cmd_dict.items():   
            results_dir = f'{args.dir}/{jobname}/work'  
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)   
            cmd = cmd + f' --dir {results_dir}'
            print(f'\n Running job {n} of {njobs}:\n{cmd}')
            res = subprocess.run(cmd, capture_output=True, text=True)
            #os.system(cmd)
            result = res.stdout
            output_per_job[jobame] = result
            n = n + 1

    df = pd.DataFrame({'jobname': output_per_job.keys(), 'output': output_per_job.values()})
    fname = f'{args.dir}/outputs.csv'
    print(f'Saving {fname}')
    df.to_csv(fname, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--parallel", type=bool, default=False)
    parser.add_argument("--machine", type=str, default="local", choices=["local", "cpu", "A10G"])

    args = parser.parse_args()
    main(args)