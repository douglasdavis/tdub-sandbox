#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import PosixPath
import json

from tdub import setup_logging
from tdub.frames import drop_cols
from tdub.train import prepare_from_root, single_training
from tdub.utils import quick_files, get_avoids, get_features, augment_features

DESCRIPTION = ""
EXECUTABLE = str(PosixPath(__file__).resolve())
BNL_CONDOR_HEADER = f"""
Universe        = vanilla
notification    = Error
notify_user     = ddavis@phy.duke.edu
GetEnv          = True
Executable      = {EXECUTABLE}
Output          = /tmp/ddavis/job.out.apply-gennpy.$(cluster).$(process)
Error           = /tmp/ddavis/job.err.apply-gennpy.$(cluster).$(process)
Log             = /tmp/ddavis/log.$(cluster).$(process)
request_memory  = 2.0G
"""


def get_args():
    # fmt: off
    parser = ArgumentParser(description=DESCRIPTION)
    action_sp = parser.add_subparsers(dest="action", help="")
    scan_p = action_sp.add_parser("scan", help="prepare a parameter scan")
    scan_p.add_argument("-c", "--config", type=str, help="json, configuration", required=True)
    scan_p.add_argument("-d", "--data-dir", type=str, help="directory containing data files", required=True)
    scan_p.add_argument("-r", "--region", type=str, help="analysis region", required=True)
    scan_p.add_argument("-n", "--nlo-method", type=str, choices=["DR", "DS"], default="DR", help="NLO method")
    scan_p.add_argument("-o", "--out-dir", type=str, help="output directory", required=True)
    single_p = action_sp.add_parser("single", help="single training round")
    single_p.add_argument("-d", "--data-dir", type=str, help="directory containing data files", required=True)
    single_p.add_argument("-r", "--region", type=str, help="analysis region", required=True)
    single_p.add_argument("-n", "--nlo-method", type=str, choices=["DR", "DS"], default="DR", help="NLO method")
    single_p.add_argument("-o", "--out-dir", type=str, help="output directory", required=True)
    single_p.add_argument("--learning-rate", type=float, required=True)
    single_p.add_argument("--num-leaves", type=int, required=True)
    single_p.add_argument("--min-child-samples", type=int, required=True)
    single_p.add_argument("--max-depth", type=int, required=True)
    single_p.add_argument("--n-estimators", type=int, default=500)
    single_p.add_argument("--early-stopping-rounds", type=int, default=10)
    # fmt: on
    return (parser.parse_args(), parser)


def single(args):
    qf = quick_files(args.data_dir)
    df, y, w = prepare_from_root(
        qf[f"tW_{args.nlo_method}"], qf["ttbar"], args.region, weight_mean=1.0,
    )
    drop_cols(df, *get_avoids(args.region))
    params = dict(
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        max_depth=args.max_depth,
    )
    sr = single_training(
        df, y, w, params, args.out_dir, early_stopping_rounds=args.early_stopping_rounds
    )
    return sr


def scan(args):
    with open(args.config, "r") as f:
        pd = json.load(f)
    p = PosixPath(args.out_dir).resolve()
    p.mkdir(exist_ok=False)
    pname = str(p)
    runs = []
    i = 0
    for max_depth in pd.get("max_depth"):
        for num_leaves in pd.get("num_leaves"):
            for learning_rate in pd.get("learning_rate"):
                for min_child_samples in pd.get("min_child_samples"):
                    suffix = "{}-{}-{}-{}".format(
                        learning_rate, num_leaves, min_child_samples, max_depth
                    )
                    arglist = (
                        "-d {} "
                        "-o {}/res{:04d}_{} "
                        "-r {} "
                        "-n {} "
                        "--learning-rate {} "
                        "--num-leaves {} "
                        "--min-child-samples {} "
                        "--max-depth {} "
                    ).format(
                        args.data_dir,
                        pname,
                        i,
                        suffix,
                        args.region,
                        args.nlo_method,
                        learning_rate,
                        num_leaves,
                        min_child_samples,
                        max_depth,
                    )
                    runs.append(arglist)
                    i += 1
    output_script = PosixPath("submit.condor")
    with output_script.open("w") as f:
        print(BNL_CONDOR_HEADER, file=f)
        for run in runs:
            print(f"Arguments = single {run}\nQueue\n\n", file=f)

def main():
    augment_features("2j1b", ["deltaR_lep1lep2_jet1jet2", "deltapT_lep1_jet1"])

    args, parser = get_args()

    if args.action == "single":
        single(args)

    if args.action == "scan":
        scan(args)


if __name__ == "__main__":
    setup_logging()
    main()
