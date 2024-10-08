from lightning_sdk import Studio
import argparse

from bong.agents import AGENT_NAMES
from job_utils import make_cmd_dict_for_flag_crossproduct


def main(args):
    cmd_dict = make_cmd_dict_for_flag_crossproduct(args)

    studio = Studio()
    studio.install_plugin("jobs")
    job_plugin = studio.installed_plugins["jobs"]

    for jobname, cmd in cmd_dict.items():
        # output_dir = make_results_dirname(jobname, args.parallel)
        # cmd = cmd + f' --dir {output_dir}' # not needed
        print("queuing job", jobname, "to run", cmd)
        job_plugin.run(cmd, name=jobname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", type=str, default="", help="directory in which to store jobs.csv"
    )
    parser.add_argument("--parallel", type=bool, default=True)

    # Data parameters
    parser.add_argument("--dataset", type=str, default="linreg")
    parser.add_argument("--key", type=int, default=0)
    parser.add_argument("--ntrain", type=int, default=500)
    parser.add_argument("--nval", type=int, default=500)
    parser.add_argument("--ntest", type=int, default=500)
    parser.add_argument("--data_dim", type=int, default=10)
    parser.add_argument("--emission_noise", type=float, default=1.0)

    parser.add_argument(
        "--agents",
        type=str,
        nargs="+",
        default=["bong-fc", "blr-fc"],
        choices=AGENT_NAMES,
    )
    parser.add_argument("--lrs", type=float, nargs="+", default=[0.01, 0.05])
    parser.add_argument("--niters", type=int, nargs="+", default=[10, 100])
    parser.add_argument("--nsamples", type=int, nargs="+", default=[10])

    args = parser.parse_args()

    main(args)

"""
python run_parallel.py  --lrs 0.001 0.01 --agents bong-fc blr-fc --dir ~/jobs/foo
"""
