#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import PosixPath
import logging
import json

from tdub import setup_logging
from tdub.frames import drop_cols
from tdub.train import (
    prepare_from_root,
    single_training,
    folded_training,
    SingleTrainingResult,
)
from tdub.utils import quick_files, get_avoids


import click

setup_logging()
log = logging.getLogger("hopt.py")


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

@click.group(context_settings=dict(max_content_width=92))
def cli():
    pass


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
    scan_p.add_argument("-x", "--extra-selection", type=str, help="input file listing extra selections before training")
    scan_p.add_argument("-t", "--use-tptrw", action="store_true", help="use top pt reweighting")
    scan_p.add_argument("-e", "--early-stopping-rounds", type=int, help="classifier early stopping rounds")
    single_p = action_sp.add_parser("single", help="single training round")
    single_p.add_argument("-d", "--data-dir", type=str, help="directory containing data files", required=True)
    single_p.add_argument("-r", "--region", type=str, help="analysis region", required=True)
    single_p.add_argument("-n", "--nlo-method", type=str, choices=["DR", "DS"], default="DR", help="NLO method")
    single_p.add_argument("-o", "--out-dir", type=str, help="output directory", required=True)
    single_p.add_argument("-x", "--extra-selection", type=str, help="input file listing extra selections before training")
    single_p.add_argument("-t", "--use-tptrw", action="store_true", help="use top pt reweighting")
    single_p.add_argument("--early-stopping-rounds", type=int, help="early stopping rounds")
    single_p.add_argument("--learning-rate", type=float, required=True)
    single_p.add_argument("--num-leaves", type=int, required=True)
    single_p.add_argument("--min-child-samples", type=int, required=True)
    single_p.add_argument("--max-depth", type=int, required=True)
    single_p.add_argument("--n-estimators", type=int, required=True)
    check_p = action_sp.add_parser("check", help="check results")
    check_p.add_argument("-d", "--direc", type=str, help="directory containing results")
    check_p.add_argument("-p", "--print", action="store_true", dest="prnt", help="print results")
    check_p.add_argument("-n", "--n-res", type=int, default=-1, help="number of top results to print")
    fold_p = action_sp.add_parser("fold", help="folded training")
    fold_p.add_argument("-s", "--scan-dir", type=str, help="scan step's output directory")
    fold_p.add_argument("-d", "--data-dir", type=str, help="directory containing data files", required=True)
    fold_p.add_argument("-o", "--out-dir", type=str, help="directory to save output", required=True)
    fold_p.add_argument("-t", "--use-tptrw", action="store_true", help="use top pt reweighting")
    fold_p.add_argument("-r", "--random-seed", type=int, default=414, help="random seed for folding")
    fold_p.add_argument("-n", "--n-splits", type=int, default=3, help="number of splits for folding")

    # fmt: on
    return (parser.parse_args(), parser)


@cli.command("check2", context_settings=dict(max_content_width=92))
@click.argument("direc", type=str)
@click.option("-p", "--print-top", is_flag=True)
@click.option("-n", "--n-res", type=int, default=-1)
def check2(direc, print_top, n_res):
    """Check the results of a parameter scan."""
    results = []
    top_dir = PosixPath(direc)
    for resdir in top_dir.iterdir():
        if resdir.name == "logs" or not resdir.is_dir():
            continue
        summary_file = resdir / "summary.json"
        if not summary_file.exists():
            continue
        with summary_file.open("r") as f:
            summary = json.load(f)
            if summary["bad_ks"]:
                continue
            res = SingleTrainingResult(**summary)
            res.directory = resdir.name
            res.summary = summary
            results.append(res)

    auc_sr = sorted(results, key=lambda r: -r.auc)
    ks_pvalue_sr = sorted(results, key=lambda r: -r.ks_pvalue_sig)
    max_auc_rounded = str(round(auc_sr[0].auc, 2))

    potentials = []
    for res in ks_pvalue_sr:
        curauc = str(round(res.auc, 2))
        if curauc == max_auc_rounded and res.ks_pvalue_bkg > 0.95:
            potentials.append(res)
        if len(potentials) > 15:
            break

    best_res = potentials[0]
    print(best_res)
    print(best_res.summary)
    print(best_res.directory)

    for result in potentials:
        print(result)

    with open(top_dir / "summary.json", "w") as f:
        json.dump(potentials[0].summary, f, indent=4)


def check(args):
    results = []
    top_dir = PosixPath(args.direc)
    for resdir in top_dir.iterdir():
        if resdir.name == "logs" or not resdir.is_dir():
            continue
        summary_file = resdir / "summary.json"
        if not summary_file.exists():
            continue
        with summary_file.open("r") as f:
            summary = json.load(f)
            if summary["bad_ks"]:
                continue
            res = SingleTrainingResult(**summary)
            res.directory = resdir.name
            res.summary = summary
            results.append(res)

    auc_sr = sorted(results, key=lambda r: -r.auc)
    ks_pvalue_sr = sorted(results, key=lambda r: -r.ks_pvalue_sig)
    max_auc_rounded = str(round(auc_sr[0].auc, 2))

    potentials = []
    for res in ks_pvalue_sr:
        curauc = str(round(res.auc, 2))
        if curauc == max_auc_rounded and res.ks_pvalue_bkg > 0.95:
            potentials.append(res)
        if len(potentials) > 15:
            break

    best_res = potentials[0]
    print(best_res)
    print(best_res.summary)
    print(best_res.directory)

    for result in potentials:
        print(result)

    with open(top_dir / "summary.json", "w") as f:
        json.dump(potentials[0].summary, f, indent=4)

    # if args.prnt:
    #     for r in ks_sorted_results:
    #         print(r.directory, r)

    # nresults = len(auc_sorted_results)
    # if args.prnt:
    #     if args.n_res > nresults:
    #         for r in auc_sorted_results:
    #             print(r.directory, r)
    #     else:
    #         for r in auc_sorted_results[: args.n_res]:
    #             print(r.directory, r)

    # auc_sorted_results[0].summary["dir"] = auc_sorted_results[0].directory
    # with open(top_dir / "summary.json", "w") as f:
    #     json.dump(auc_sorted_results[0].summary, f, indent=4)


def single(args):
    qf = quick_files(args.data_dir)
    extra_sel = args.extra_selection
    if extra_sel:
        extra_sel = PosixPath(extra_sel).read_text().strip()
    df, y, w = prepare_from_root(
        qf[f"tW_{args.nlo_method}"],
        qf["ttbar"],
        args.region,
        weight_mean=1.0,
        extra_selection=extra_sel,
        use_tptrw=args.use_tptrw,
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
    best_iteration = summary["best_iteration"]
    if best_iteration > 0:
        summary["all_params"]["n_estimators"] = best_iteration
    region = summary["region"]
    branches = summary["features"]
    qf = quick_files(args.data_dir)
    df, y, w = prepare_from_root(
        qf[f"tW_{nlo_method}"],
        qf["ttbar"],
        region,
        branches=branches,
        weight_mean=1.0,
        use_tptrw=args.use_tptrw,
    )
    drop_cols(df, *get_avoids(region))
    folded_training(
        df,
        y,
        w,
        summary["all_params"],
        {"verbose": 10},
        args.out_dir,
        summary["region"],
        kfold_kw={
            "n_splits": args.n_splits,
            "shuffle": True,
            "random_state": args.random_seed,
        },
    )
    return 0

@cli.command("scan2", context_settings=dict(max_content_width=140))
@click.argument("config", type=str)
@click.argument("data-dir", type=str)
@click.option("-r", "--region", type=str, required=True, help="the region to train on")
@click.option("-o", "--out-dir", type=str, required=True, help="output directory name")
@click.option("-n", "--nlo-method", type=str, default="DR", help="tW simluation NLO method", show_default=True)
@click.option("-s", "--script-name", type=str, default="hopt.scan.REGION.sub", help="output script name", show_default=True)
@click.option("-x", "--extra-selection", type=str, help="extra selection string")
@click.option("-t", "--use-tptrw", is_flag=True, help="apply top pt reweighting")
@click.option("-e", "--early-stopping-rounds", type=int, help="number of early stopping rounds")
def scan2(config, data_dir, region, out_dir, nlo_method, script_name, extra_selection, use_tptrw, early_stopping_rounds):
    """Generate a condor submission script to execute a hyperparameter scan.

    The scan parameters are defined in the CONFIG file, and the data
    to use is in the DATA_DIR. Example:

    $ hopt.py scan conf.json /data/path -r 2j1b -e 10 -o scan_2j1b

    """
    with open(config, "r") as f:
        pd = json.load(f)
    p = PosixPath(out_dir).resolve()
    p.mkdir(exist_ok=False)
    (p / "logs").mkdir(exist_ok=False)
    pname = str(p)
    runs = []
    i = 0
    extra_sel = extra_selection
    if extra_sel is None:
        extra_sel = "_NONE"
    else:
        extra_sel = str(PosixPath(extra_sel).resolve())
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
                            "{}"
                            "-d {} "
                            "-o {}/res{:04d}_{} "
                            "-r {} "
                            "-n {} "
                            "-x {} "
                            "--learning-rate {} "
                            "--num-leaves {} "
                            "--n-estimators {} "
                            "--min-child-samples {} "
                            "--max-depth {} "
                            "--early-stopping-rounds {} "
                        ).format(
                            "-t " if use_tptrw else "",
                            data_dir,
                            pname,
                            i,
                            suffix,
                            region,
                            nlo_method,
                            extra_sel,
                            learning_rate,
                            num_leaves,
                            n_estimators,
                            min_child_samples,
                            max_depth,
                            early_stopping_rounds
                        )
                        arglist = arglist.replace("-x _NONE ", "")
                        runs.append(arglist)
                        i += 1
    log.info(f"prepared {len(runs)} jobs for submission")
    output_script_name = script_name.replace("REGION", region)
    output_script = PosixPath(output_script_name)
    with output_script.open("w") as f:
        print(BNL_CONDOR_HEADER.format(exe=EXECUTABLE, out_dir=pname), file=f)
        for run in runs:
            print(f"Arguments = single {run}\nQueue\n\n", file=f)




def scan(args):
    with open(args.config, "r") as f:
        pd = json.load(f)
    p = PosixPath(args.out_dir).resolve()
    p.mkdir(exist_ok=False)
    (p / "logs").mkdir(exist_ok=False)
    pname = str(p)
    runs = []
    i = 0
    extra_sel = args.extra_selection
    if extra_sel is None:
        extra_sel = "_NONE"
    else:
        extra_sel = str(PosixPath(extra_sel).resolve())
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
                            "{}"
                            "-d {} "
                            "-o {}/res{:04d}_{} "
                            "-r {} "
                            "-n {} "
                            "-x {} "
                            "--learning-rate {} "
                            "--num-leaves {} "
                            "--n-estimators {} "
                            "--min-child-samples {} "
                            "--max-depth {} "
                            "--early-stopping-rounds {} "
                        ).format(
                            "-t " if args.use_tptrw else "",
                            args.data_dir,
                            pname,
                            i,
                            suffix,
                            args.region,
                            args.nlo_method,
                            extra_sel,
                            learning_rate,
                            num_leaves,
                            n_estimators,
                            min_child_samples,
                            max_depth,
                            args.early_stopping_rounds
                        )
                        arglist = arglist.replace("-x _NONE ", "")
                        runs.append(arglist)
                        i += 1
    log.info(f"prepared {len(runs)} jobs for submission")
    output_script_name = args.script_name.replace("REGION", args.region)
    output_script = PosixPath(output_script_name)
    with output_script.open("w") as f:
        print(BNL_CONDOR_HEADER.format(exe=EXECUTABLE, out_dir=pname), file=f)
        for run in runs:
            print(f"Arguments = single {run}\nQueue\n\n", file=f)


def main():
    pass
    # augment_features("2j1b", ["deltaR_lep1lep2_jet1jet2", "deltapT_lep1_jet1"])

    #args, parser = get_args()

    #if args.action == "single":
    #    single(args)

    #if args.action == "scan":
    #    scan(args)

    #if args.action == "check":
    #    check(args)

    #if args.action == "fold":
    #    fold(args)


if __name__ == "__main__":
    import tdub.constants
    tdub.constants.AVOID_IN_CLF_1j1b = []
    tdub.constants.AVOID_IN_CLF_2j1b = []
    tdub.constants.AVOID_IN_CLF_2j2b = []
    cli()
