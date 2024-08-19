import argparse
import os
import pandas as pd
from pathlib import Path
import json


def main(args):
    path = Path(args.dir)
    results_dir = str(path)
    fname = f"{results_dir}/jobs.csv"
    df = pd.read_csv(fname)
    jobnames = df["jobname"]
    jobs_dir = "/teamspace/jobs"  # when run in parallel mode, all results written here.

    fname = f"{results_dir}/args.json"
    with open(fname, "r") as json_file:
        jobargs = json.load(json_file)
    ntrials = jobargs["ntrials"]
    print("ntrials", ntrials)

    for job in jobnames:
        for i in range(ntrials):
            jobname = f"{job}-trial{i}"
            src = f"{jobs_dir}/{jobname}/work"
            src_path = Path(src)
            if not src_path.exists():
                print(f"Cannot find {str(src_path)}, skipping")
                continue

            dst = f"{results_dir}/jobs/{jobname}"
            dst_path = Path(dst)
            print(f"\n Creating {str(dst_path)}")
            dst_path.mkdir(parents=True, exist_ok=True)

            # chnage permissions so we can later on delete the copied files
            cmd = f"chmod ugo+rwx {dst}"
            os.system(cmd)

            fnames = ["results.csv", "args.json"]
            for fname in fnames:
                cmd = f"cp -r {src}/{fname} {dst}/{fname}"
                # print(f'Running {cmd}')
                try:
                    os.system(cmd)
                except Exception as e:
                    print(f"Error {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="")

    args = parser.parse_args()
    main(args)
