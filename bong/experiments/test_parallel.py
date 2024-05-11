import argparse
from pathlib import Path
import os

cwd = Path(os.getcwd())
root = cwd
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

def main(parallel=False):
    if parallel:
        from lightning_sdk import Studio, Machine
        print('parallel')
        studio = Studio()
        job_plugin = studio.installed_plugins['jobs']
    else:
        print('serial')
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=bool, default=False)
    args = parser.parse_args()

    main_name = 'main'
    agent = 'agent'
    nsample = 100
    cmd = (f'python {main_name} --agent {agent}'
        f'--nsample {nsample}')
    print(cmd)

    print(script_path)
    print(script_dir)
    main(args.parallel)
