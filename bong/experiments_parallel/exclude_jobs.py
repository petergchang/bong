import argparse
import os
import pandas as pd


def main(args):
    # dir = '/teamspace/studios/this_studio/jobs/timingdlr'
    src = f"{args.dir}/jobs.csv"
    dst = f"{args.dir}/jobs_all.csv"
    cmd = f"cp {src} {dst}"  # keep copy of original!
    print(cmd)
    os.system(cmd)

    df = pd.read_csv(f"{args.dir}/jobs.csv")
    # edit me!
    # condition = (df['agent'] == "bbb_fc") & (df['ef'] == 0) & (df['linplugin'] == 0)
    condition = (df["agent"] == "blr_dlr") & (df["linplugin"] == 0)

    indices_to_drop = df[condition].index
    df_filtered = df.drop(indices_to_drop)

    n_before, n_after = len(df), len(df_filtered)
    print(f"Reduced from {n_before} to {n_after} jobs")

    df_filtered.to_csv(f"{args.dir}/jobs.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    args = parser.parse_args()
    main(args)
