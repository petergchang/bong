

import argparse
import os
import itertools
import pandas as pd
from pathlib import Path
import os
import datetime

from bong.agents import AGENT_DICT, AGENT_NAMES, make_agent_name_from_parts
from bong.util import safestr, make_neuron_str, unmake_neuron_str, make_file_with_timestamp
from plot_utils import plot_results, extract_results_from_files, extract_metrics_from_files

cwd = Path(os.getcwd())
root = cwd
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)


def make_dataset_dirname(args):
    neurons_str = make_neuron_str(args.dgp_neurons)
    name = f'{args.dataset}-D{args.data_dim}-{args.dgp_type}_{neurons_str}'
    return name

def make_model_dirname(args):
    neurons_str = make_neuron_str(args.model_neurons)
    name = f'{args.model_type}_{neurons_str}'
    return name



def foo(lst):
    if len(lst)>1:
        return "Any"
    else:
        x = lst[0]
        if isinstance(x, float):
            return safestr(x)
        else:
            return x

def make_agent_dirname(args):
    # Example output: A:bong-P:fc-Lin:0-LR:0_05-I:10-MC:10-EF:0-R:1
    parts = {
        'A': foo(args.algo_list),
        'P': foo(args.param_list),
        'Lin': foo(args.lin_list),
        'LR': foo(args.lr_list),
        'IT': foo(args.niter_list),
        'MC': foo(args.nsample_list),
        'EF': foo(args.ef_list),
        'R': foo(args.rank_list)
        }
    parts_str = [f"{name}:{val}" for (name, val) in parts.items()]
    name = "-".join(parts_str)
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
        f' --niter {niter} --nsample {nsample} --linplugin {linplugin}'
        f' --ef {ef} --rank {rank}'
        f' --model_type {model_type} --model_neurons_str {model_neurons_str}'
        f' --dataset {dataset} --data_dim {data_dim}'
        f' --dgp_type {dgp_type} --dgp_neurons_str {dgp_neurons_str}'
        f' --ntrain {ntrain}'
    )
    return cmd




def make_df_for_flag_crossproduct(jobprefix, 
        algo_list, param_list, lin_list,
        lr_list, niter_list, nsample_list, ef_list, rank_list):
    args_list = []
    for algo in algo_list:
        for param in param_list:
            for lin in lin_list:
                agent = make_agent_name_from_parts(algo, param, lin)
                props = AGENT_DICT[agent]
                for lr in lr_list:
                    for niter in niter_list:
                        for nsample in nsample_list:
                            for ef in ef_list:
                                for rank in rank_list:
                                    args = extract_optional_agent_args(props, lr, niter, nsample, ef, rank)
                                    args['agent'] = agent
                                    args_list.append(args)
    df = pd.DataFrame(args_list)
    df = df.drop_duplicates()
    N = len(df)
    jobnames = [f'{jobprefix}-{i}' for i in range(N)] 
    df['jobname'] = jobnames
    return df


def make_and_save_results(args, path):
    # Make sure we can save results before doing any compute
    results_dir = str(path)
    print(f'Saving job results in {results_dir}')
    path.mkdir(parents=True, exist_ok=True)

    df_flags = make_df_for_flag_crossproduct(
        args.job_prefix, args.algo_list, args.param_list, args.lin_list,
        args.lr_list, args.niter_list, args.nsample_list,
        args.ef_list, args.rank_list)

    # for flags that are shared across all jobs, we create extra columns (duplicated across rows)
    df_flags['dataset'] = args.dataset
    df_flags['data_dim'] = args.data_dim
    df_flags['dgp_type'] = args.dgp_type
    df_flags['dgp_neurons_str'] = make_neuron_str(args.dgp_neurons)
    df_flags['model_type'] = args.model_type
    df_flags['model_neurons_str'] = make_neuron_str(args.model_neurons)
    df_flags['ntrain'] = args.ntrain


    cmd_dict = {}
    cmd_list = []
    for index, row in df_flags.iterrows():
        cmd = make_unix_cmd_given_flags(
            row.agent, row.lr, row.niter, row.nsample,
            row.linplugin, row.ef, row.dlr_rank, 
            row.model_type, row.model_neurons_str, 
            row.dataset, row.data_dim, 
            row.dgp_type, row.dgp_neurons_str, row.ntrain)
        cmd_dict[row.jobname] = cmd
        cmd_list.append(cmd)
    #df_flags['cmd'] = cmd_list
    

    fname = Path(path, "jobs.csv")
    df_flags.to_csv(fname, index=False) 

    cmds = [{'jobname': key, 'command': value} for key, value in cmd_dict.items()]
    df_cmds = pd.DataFrame(cmds)
    fname = Path(path, "cmds.csv")
    df_cmds.to_csv(fname, index=False)
    njobs = len(df_cmds)
    print(f'Running {njobs} jobs, please be patient')


    if args.parallel:
        from lightning_sdk import Studio, Machine
        studio = Studio()
        studio.install_plugin('jobs')
        job_plugin = studio.installed_plugins['jobs']
        jobs_dir = '/teamspace/jobs'
        #print(f'Will store results in /teamspace/jobs/[jobname]/work')

        n = 0
        for jobname, cmd in cmd_dict.items():
            print(f'\n Queuing job {n} of {njobs}:\n{cmd}')
            output_dir = f'{jobs_dir}/{jobname}/work' 
            if args.gpu == 'None':
                job_plugin.run(cmd, name=jobname) # run on local VM
            elif args.gpu == 'A10G':
                job_plugin.run(cmd, machine=Machine.A10G, name=jobname)
            n = n + 1

    else:
        jobs_dir = results_dir
        n = 0
        for jobname, cmd in cmd_dict.items():   
            output_dir = f'{jobs_dir}/{jobname}/work'  
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)   
            cmd = cmd + f' --dir {output_dir}'
            print(f'\n Running job {n} of {njobs}:\n{cmd}')
            os.system(cmd)
            n = n + 1
    

def copy_results(args, path):
    results_dir = str(path)
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

def main(args):
    if args.dir == "":
        data_dirname = make_dataset_dirname(args)
        model_dirname = make_model_dirname(args)
        agent_dirname = make_agent_dirname(args)
        path = Path(args.rootdir, data_dirname, model_dirname, agent_dirname)
    else:
        path = Path(args.dir)
    results_dir = str(path)

    if args.plot:
        print(f'Writing plots to {results_dir}')
        make_file_with_timestamp(results_dir)
        metrics = extract_metrics_from_files(results_dir)
        for metric in metrics:
            results = extract_results_from_files(results_dir,  metric)
            fig, ax = plot_results(results,  metric)
            fname = f"{results_dir}/{metric}.png"
            print(f'Saving figure to {fname}')
            fig.savefig(fname, bbox_inches='tight', dpi=300)
        return

    if args.copy:
        copy_results(args, path)
        return

    # Not copy, not plot, so do some real work
    make_and_save_results(args, path)
   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir", type=str, default="/teamspace/studios/this_studio/jobs") 
    parser.add_argument("--dir", type=str, default="")
    parser.add_argument("--job_prefix", type=str, default="job")
    parser.add_argument("--parallel", type=bool, default=False)
    parser.add_argument("--gpu", type=str, default="None", choices=["None", "A10G"])
    parser.add_argument("--plot", type=bool, default=False)
    parser.add_argument("--copy", type=bool, default=False)


    # Data parameters
    parser.add_argument("--dataset", type=str, default="reg")  
    parser.add_argument("--data_dim", type=int,  default=10)
    parser.add_argument("--dgp_type", type=str, default="lin") # or mlp
    parser.add_argument("--dgp_neurons", type=int, nargs="+", default=[20,20,1]) 
    parser.add_argument("--ntrain", type=int,  default=500)

    
    # Agent parameters
    parser.add_argument("--algo_list", type=str, nargs="+", default=["bong"])
    parser.add_argument("--param_list", type=str, nargs="+", default=["fc"])
    parser.add_argument("--lin_list", type=int, nargs="+", default=[0])
    parser.add_argument("--lr_list", type=float, nargs="+", default=[0.01])
    parser.add_argument("--niter_list", type=int, nargs="+", default=[10])
    parser.add_argument("--nsample_list", type=int, nargs="+", default=[10])
    parser.add_argument("--ef_list", type=int, nargs="+", default=[1])
    parser.add_argument("--rank_list", type=int, nargs="+", default=[10])
    parser.add_argument("--model_type", type=str, default="lin") # or mlp
    parser.add_argument("--model_neurons", type=int, nargs="+", default=[10, 10, 1])



    args = parser.parse_args()
    print(args)
    
    main(args)

