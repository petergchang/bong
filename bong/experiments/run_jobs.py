

import argparse
import os
import itertools
import pandas as pd
from pathlib import Path
import os

from bong.agents import AGENT_DICT, AGENT_NAMES
from bong.util import safestr
from plot_utils import plot_results_from_files

cwd = Path(os.getcwd())
root = cwd
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)


def make_dataset_dirname(args):
    #if hasattr(args, 'data_dim'):
    #    data_dim = args.data_dim
    if args.dataset == "linreg":
        name = f'linreg-dim{args.data_dim}-key{args.data_key}'
    elif args.dataset == "mlpreg":
        name = f'mlpreg-dim{args.data_dim}-mlp{args.data_neurons}-key{args.data_key}'
    else:
        raise Exception(f'Unknown dataset {args.dataset}')
    return name


def make_agent_dirname(args):
    parts = []
    if hasattr(args, 'algo_list'):
        if len(args.algo_list)>1:
            s =  "Any"
        else:
            s = args.algo_list[0]
    else:
        s = args.algo
    parts.append(f"A:{s}")

    if hasattr(args, 'param_list'):
        if len(args.param_list)>1:
            s =  "Any"
        else:
            s = args.param_list[0]
    else:
        s = args.param
    parts.append(f"P:{s}")

    if hasattr(args, 'lin_list'):
        if len(args.lin_list)>1:
            s =  "Any"
        else:
            s = args.lin_list[0]
    else:
        s = args.lin
    parts.append(f"Lin:{s}")

    if hasattr(args, 'lr_list'):
        if len(args.lr_list)>1:
            s =  "Any"
        else:
            s = safestr(args.lr_list[0])
    else:
        s = safestr(args.lr)
    parts.append(f"LR:{s}")

    if hasattr(args, 'niter_list'):
        if len(args.niter_list)>1:
            s =  "Any"
        else:
            s = args.niter_list[0]
    else:
        s = args.niter
    parts.append(f"I:{s}")

    if hasattr(args, 'nsample_list'):
        if len(args.nsample_list)>1:
            s =  "Any"
        else:
            s = args.nsample_list[0]
    else:
        s = args.nsample
    parts.append(f"MC:{s}")

    if hasattr(args, 'ef_list'):
        if len(args.ef_list)>1:
            s =  "Any"
        else:
            s = args.ef_list[0]
    else:
        s = args.ef
    parts.append(f"EF:{s}")

    if hasattr(args, 'rank_list'):
        if len(args.rank_list)>1:
            s =  "Any"
        else:
            s = args.rank_list[0]
    else:
        s = args.rank
    parts.append(f"R:{s}")

    if hasattr(args, 'model_neurons_list'):
        if len(args.model_neurons_list)>1:
            s =  "Any"
        else:
            s = args.model_neurons_list[0]
    else:
        s = args.model_neurons
    parts.append(f"MLP:{s}")

    name = "-".join(parts)
    return name



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


def make_df_for_flag_crossproduct(jobprefix, 
        algo_list, param_list, lin_list,
        lr_list, niter_list, nsample_list, ef_list, model_neurons_list, rank_list):
    args_list = []
    for algo in algo_list:
        for param in param_list:
            for lin in lin_list:
                if lin:
                    agent = f'{algo}_{param}_lin'
                else:
                    agent = f'{algo}_{param}'
                props = AGENT_DICT[agent]
                for lr in lr_list:
                    for niter in niter_list:
                        for nsample in nsample_list:
                            for ef in ef_list:
                                for model_neurons in model_neurons_list:
                                    for rank in rank_list:
                                        args = extract_optional_agent_args(props, lr, niter, nsample, ef, rank)
                                        args['agent'] = agent
                                        args['model_neurons'] = model_neurons
                                        args_list.append(args)
    df = pd.DataFrame(args_list)
    df = df.drop_duplicates()
    N = len(df)
    #jobnames = [f'job-{i}' for i in range(N)] 
    jobnames = [f'{jobprefix}-{i}' for i in range(N)] 
    df['jobname'] = jobnames
    #dirs = [make_results_dirname(j, parallel) for j in jobnames]
    #df['results_dir'] = dirs
    return df



def main(args):
    if args.dir == "":
        data_dirname = make_dataset_dirname(args)
        agent_dirname = make_agent_dirname(args)
        path = Path(args.rootdir, data_dirname, agent_dirname)
    else:
        path = Path(args.dir)
    results_dir = str(path)

    if args.plot:
        print(f'Writing plots to {results_dir}')
        for metric in ['kl', 'nll', 'nlpd']:
            plot_results_from_files(results_dir,  metric, save_fig=True)
        return

    if args.copy:
        fname = f"{results_dir}/jobs.csv"
        df = pd.read_csv(fname)
        jobnames = df['jobname']
        jobs_dir = '/teamspace/jobs' # when run in parallel mode
        for job in jobnames:
            src = f'{jobs_dir}/{job}/work'
            dst = f'{results_dir}/{job}/work'
            dst_path = Path(dst)
            print(f'\n Creating {str(dst_path)}')
            dst_path.mkdir(parents=True, exist_ok=True)

            # chnage permissions so we can delete the copied files
            cmd = f'chmod ugo+rwx {dst}'
            print(f'Running {cmd}')
            os.system(cmd)

            fnames = ['results.csv', 'args.json']
            for fname in fnames:
                cmd = f'cp -r {src}/{fname} {dst}/{fname}'
                print(f'Running {cmd}')
                os.system(cmd)
        return

    # Create new results
    print(f'Creating {str(path)}')
    path.mkdir(parents=True, exist_ok=True)

    df_flags = make_df_for_flag_crossproduct(
        args.job_prefix, args.algo_list, args.param_list, args.lin_list,
        args.lr_list, args.niter_list, args.nsample_list,
        args.ef_list, args.model_neurons_list, args.rank_list)
        #args.data_dim_list, args.data_key_list)

    # for flags that are shared across all jobs, we create extra columns (duplicated across rows)
    df_flags['dataset'] = args.dataset
    df_flags['data_dim'] = args.data_dim
    df_flags['data_key'] = args.data_key

    cmd_dict = {}
    cmd_list = []
    for index, row in df_flags.iterrows():
        cmd = make_unix_cmd_given_flags(
            row.agent, row.lr, row.niter, row.nsample,
            row.linplugin, row.ef, row.model_neurons, row.dlr_rank, # rank is a reserved word in pandas
            row.dataset, row.data_dim, row.data_key)
        cmd_dict[row.jobname] = cmd
        cmd_list.append(cmd)
    df_flags['cmd'] = cmd_list
    

    # Store csv containing all the flags/commands that are being executed
    fname = Path(path, "jobs.csv")
    print("Saving to", fname)
    df_flags.to_csv(fname, index=False) 

    cmds = [{'jobname': key, 'command': value} for key, value in cmd_dict.items()]
    df_cmds = pd.DataFrame(cmds)
    fname = Path(path, "cmds.csv")
    print("Saving to", fname)
    df_cmds.to_csv(fname, index=False)


    if args.parallel:
        from lightning_sdk import Studio, Machine
        studio = Studio()
        studio.install_plugin('jobs')
        job_plugin = studio.installed_plugins['jobs']
        jobs_dir = '/teamspace/jobs'
        #print(f'Will store results in /teamspace/jobs/[jobname]/work')

        for jobname, cmd in cmd_dict.items():
            print('\n Queuing job', jobname)
            output_dir = f'{jobs_dir}/{jobname}/work' 
            print(cmd)
            print(f'saving output to {output_dir}')
            job_plugin.run(cmd, name=jobname) # run locally
            #job_plugin.run(cmd, machine=Machine.A10G_X_4, name=jobname)

    else:
        jobs_dir = results_dir
        #print(f'Storing results in {args.dir}')
        for jobname, cmd in cmd_dict.items():   
            output_dir = f'{jobs_dir}/{jobname}/work'  
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)   
            cmd = cmd + f' --dir {output_dir}'
            print('\n Running', cmd)
            #print(f'Storing results in {output_dir}')
            os.system(cmd)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir", type=str, default="/teamspace/studios/this_studio/jobs") 
    parser.add_argument("--dir", type=str, default="")
    parser.add_argument("--job_prefix", type=str, default="job")
    parser.add_argument("--parallel", type=bool, default=False)
    parser.add_argument("--plot", type=bool, default=False)
    parser.add_argument("--copy", type=bool, default=False)

    # Data parameters
    parser.add_argument("--dataset", type=str, default="linreg") # ['linreg', 'mlpreg'] 
    parser.add_argument("--data_dim", type=int,  default=10)
    parser.add_argument("--data_key", type=int,  default=0)
    #parser.add_argument("--data_dim_list", type=int, nargs="+", default=[10])
    #parser.add_argument("--data_key_list", type=int, nargs="+", default=[0])
    
    # Agent parameters
    parser.add_argument("--algo_list", type=str, nargs="+", default=["bong"])
    parser.add_argument("--param_list", type=str, nargs="+", default=["fc"])
    parser.add_argument("--lin_list", type=int, nargs="+", default=[0])
    parser.add_argument("--lr_list", type=float, nargs="+", default=[0.01])
    parser.add_argument("--niter_list", type=int, nargs="+", default=[10])
    parser.add_argument("--nsample_list", type=int, nargs="+", default=[10])
    parser.add_argument("--ef_list", type=int, nargs="+", default=[1])
    parser.add_argument("--rank_list", type=int, nargs="+", default=[10])
    parser.add_argument("--model_neurons_list", type=str, nargs="+",  default=["1"]) # 10-10-1


    args = parser.parse_args()
    print(args)
    
    main(args)

