import argparse
from pathlib import Path
import os

cwd = Path(os.getcwd())
root = cwd
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)


def main(parallel=False):
    print(args.lr_list)
    print(args.agent_list)
    print(args.data)
    print(args.data_dim)
    if parallel:
        from lightning_sdk import Studio

        print("parallel")
        studio = Studio()
        job_plugin = studio.installed_plugins["jobs"]
    else:
        print("serial")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=bool, default=False)
    parser.add_argument("--lr_list", type=float, nargs="+", default=[0.01])
    parser.add_argument("--agent_list", type=str, nargs="+", default=["foo", "bar"])
    parser.add_argument("--data", type=str, default="linreg")
    parser.add_argument("--data_dim", type=int, default=10)
    args = parser.parse_args()

    main(args.parallel)

"""
LR_LIST=(0.01 0.1)
A_LIST=("a" "b")
DATA="mlp"
DATADIM=10
python test_parallel.py   --lr_list ${LR_LIST[@]} --agent_list ${A_LIST[@]} --data $DATA --data_dim $DATADIM
"""
