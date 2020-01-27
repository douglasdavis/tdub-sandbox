#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import PosixPath
from pprint import pprint
import json

from tdub import setup_logging
from tdub.frames import drop_cols
from tdub.train import (
    prepare_from_root,
    single_training,
    folded_training,
    SingleTrainingResult,
)
from tdub.utils import quick_files, get_avoids, get_features, augment_features

DESCRIPTION = ""
EXECUTABLE = str(PosixPath(__file__).resolve())
BNL_CONDOR_HEADER = """
Universe        = vanilla
notification    = Error
notify_user     = ddavis@phy.duke.edu
GetEnv          = True
Executable      = {exe}
Output          = {out_dir}/logs/job.out.hopt.$(cluster).$(process)
Error           = {out_dir}/logs/job.err.hopt.$(cluster).$(process)
Log             = {out_dir}/logs/job.log.$(cluster).$(process)
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
    scan_p.add_argument("-s", "--script-name", type=str, default="condor.hopt.REGION.sub", help="output script name")
    single_p = action_sp.add_parser("single", help="single training round")
    single_p.add_argument("-d", "--data-dir", type=str, help="directory containing data files", required=True)
    single_p.add_argument("-r", "--region", type=str, help="analysis region", required=True)
    single_p.add_argument("-n", "--nlo-method", type=str, choices=["DR", "DS"], default="DR", help="NLO method")
    single_p.add_argument("-o", "--out-dir", type=str, help="output directory", required=True)
    single_p.add_argument("--learning-rate", type=float, required=True)
    single_p.add_argument("--num-leaves", type=int, required=True)
    single_p.add_argument("--min-child-samples", type=int, required=True)
    single_p.add_argument("--max-depth", type=int, required=True)
    single_p.add_argument("--n-estimators", type=int, required=True)
    single_p.add_argument("--early-stopping-rounds", type=int, help="early stopping rounds")
    check_p = action_sp.add_parser("check", help="check results")
    check_p.add_argument("indir", type=str, help="directory containing results")
    check_p.add_argument("-p", "--print", action="store_true", dest="prnt", help="print results")
    check_p.add_argument("-n", "--n-res", type=int, default=-1, help="number of top results to print")
    fold_p = action_sp.add_parser("fold", help="folded training")
    fold_p.add_argument("-s", "--scan-dir", type=str, help="scan step's output directory")
    fold_p.add_argument("-d", "--data-dir", type=str, help="directory containing data files", required=True)
    fold_p.add_argument("-o", "--out-dir", type=str, help="directory to save output")
    fold_p.add_argument("--seed", type=int, default=414, help="random seed for folding")
    fold_p.add_argument("--n-splits", type=int, default=3, help="number of splits for folding")
    fold_p.add_argument("--early-stopping-rounds", dest="esr", type=int, help="early stopping rounds")

    # fmt: on
    return (parser.parse_args(), parser)


def check(args):
    results = []
    top_dir = PosixPath(args.indir)
    for resdir in top_dir.iterdir():
        if resdir.name == "logs" or not resdir.is_dir():
            continue
        with open(resdir / "summary.json", "r") as f:
            summary = json.load(f)
            if summary["bad_ks"] > 0:
                continue
            res = SingleTrainingResult(auc=summary["auc"])
            res.directory = resdir.name
            res.summary = summary
            results.append(res)
    results = sorted(results, key=lambda r: -r.auc)
    results[0].summary["dir"] = results[0].directory
    with open(top_dir / "summary.json", "w") as f:
        json.dump(results[0].summary, f, indent=4)

    nresults = len(results)
    if args.prnt:
        if args.n_res > nresults:
            for r in results:
                print(r.directory, r.auc)
        else:
            for r in results[: args.n_res]:
                print(r.directory, r.auc)


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
        n_estimators=args.n_estimators,
    )
    extra_sum = {"region": args.region, "nlo_method": args.nlo_method}
    sr = single_training(
        df,
        y,
        w,
        params,
        args.out_dir,
        early_stopping_rounds=args.early_stopping_rounds,
        extra_summary_entries=extra_sum,
    )
    return sr


def fold(args):
    with open(f"{args.scan_dir}/summary.json", "r") as f:
        summary = json.load(f)
    nlo_method = summary["nlo_method"]
    region = summary["region"]

    qf = quick_files(args.data_dir)
    df, y, w = prepare_from_root(
        qf[f"tW_{nlo_method}"], qf["ttbar"], region, weight_mean=1.0,
    )
    drop_cols(df, *get_avoids(region))
    folded_training(
        df,
        y,
        w,
        summary["all_params"],
        {"verbose": 20, "early_stopping_rounds": args.esr},
        args.out_dir,
        summary["region"],
        kfold_kw={
            "n_splits": args.n_splits,
            "shuffle": True,
            "random_state": args.seed,
        },
    )
    return 0


def scan(args):
    with open(args.config, "r") as f:
        pd = json.load(f)
    p = PosixPath(args.out_dir).resolve()
    p.mkdir(exist_ok=False)
    (p / "logs").mkdir(exist_ok=False)
    pname = str(p)
    runs = []
    i = 0
    for max_depth in pd.get("max_depth"):
        for num_leaves in pd.get("num_leaves"):
            for n_estimators in pd.get("n_estimators"):
                for learning_rate in pd.get("learning_rate"):
                    for min_child_samples in pd.get("min_child_samples"):
                        suffix = "{}-{}-{}-{}-{}".format(
                            learning_rate,
                            num_leaves,
                            n_estimators,
                            min_child_samples,
                            max_depth,
                        )
                        arglist = (
                            "-d {} "
                            "-o {}/res{:04d}_{} "
                            "-r {} "
                            "-n {} "
                            "--learning-rate {} "
                            "--num-leaves {} "
                            "--n-estimators {} "
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
                            n_estimators,
                            min_child_samples,
                            max_depth,
                        )
                        runs.append(arglist)
                        i += 1
    output_script_name = args.script_name.replace("REGION", args.region)
    output_script = PosixPath(output_script_name)
    with output_script.open("w") as f:
        print(BNL_CONDOR_HEADER.format(exe=EXECUTABLE, out_dir=pname), file=f)
        for run in runs:
            print(f"Arguments = single {run}\nQueue\n\n", file=f)


def main():
    augment_features("2j1b", ["deltaR_lep1lep2_jet1jet2", "deltapT_lep1_jet1"])

    args, parser = get_args()

    if args.action == "single":
        single(args)

    if args.action == "scan":
        scan(args)

    if args.action == "check":
        check(args)

    if args.action == "fold":
        fold(args)


if __name__ == "__main__":
    setup_logging()
    main()
