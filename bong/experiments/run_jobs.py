from lightning_sdk import Studio, Machine
import argparse
import os
import itertools

def make_cmd(agent, lr, niter):
    main_name = '/teamspace/studios/this_studio/bong/bong/experiments/main.py'
    cmd = f'python {main_name} --agent {agent} --learning_rate {lr} --num_iter {niter}'
    return cmd

def main(args):
    grid_search_params = list(itertools.product(args.agent, args.learning_rate, args.num_iter))
    #grid_search_params = [(lr, agent) for lr in args.learning_rate for agent in args.agent]

    if args.parallel:
        studio = Studio()
        studio.install_plugin('jobs')
        job_plugin = studio.installed_plugins['jobs']

        for index, (agent, lr, niter) in enumerate(grid_search_params):
            cmd = make_cmd(agent, lr, niter)
            job_name = f'bong-{index}'
            print('queuing job', job_name, 'to run', cmd)
            #job_plugin.run(cmd, machine=Machine.A10G, name=job_name)
            job_plugin.run(cmd, name=job_name)
            # results stored in /teamspace/jobs/job_name/work/xxx
    else:
        for index, (agent, lr, niter) in enumerate(grid_search_params):
            cmd = make_cmd(agent, lr, niter)
            job_name = f'bong-{index}'
            output_dir = f'/teamspace/studios/this_studio/jobs/{job_name}/work'
            cmd = cmd + f'--dir {output_dir}'
            print('running', cmd)
            os.system(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--parallel", type=bool, default=False)
    parser.add_argument("--agent", type=str, nargs="+",
                        default=["fg-bong", "fg-blr"])
    parser.add_argument("--learning_rate", type=float, nargs="+", 
                    default=[1,2,3])
    parser.add_argument("--num_iter", type=int, nargs="+", 
                    default=[10,100])
  
    args = parser.parse_args()
    
    main(args)