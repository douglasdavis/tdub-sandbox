#!/usr/bin/env python3

# stdlib
from pathlib import PosixPath
import logging
import json

# third party
import click

# tdub
from tdub import setup_logging
from tdub.utils import quick_files

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


@cli.command("train-single", context_settings=dict(max_content_width=92))
@click.option("-d", "--data-dir", type=str, help="directory containing data files", required=True)
@click.option("-r", "--region", type=str, required=True, help="the region to train on")
@click.option("-o", "--out-dir", type=str, required=True, help="output directory name")
@click.option("-n", "--nlo-method", type=str, default="DR", help="tW simluation NLO method", show_default=True)
@click.option("-x", "--extra-selection", type=str, help="extra selection file")
@click.option("-t", "--use-tptrw", is_flag=True, help="apply top pt reweighting")
@click.option("-e", "--early-stopping-rounds", type=int, help="number of early stopping rounds")
@click.option("--learning-rate", type=float, required=True, help="learning_rate model parameter")
@click.option("--num-leaves", type=int, required=True, help="num_leaves model parameter")
@click.option("--min-child-samples", type=int, required=True, help="min_child_samples model parameter")
@click.option("--max-depth", type=int, required=True, help="max_depth model parameter")
@click.option("--n-estimators", type=int, required=True, help="n_estimators model parameter")
def single(data_dir, region, out_dir, nlo_method, extra_selection, use_tptrw, early_stopping_rounds,
           learning_rate, num_leaves, min_child_samples, max_depth, n_estimators):
    """Execute a single training round."""
    from tdub.train import single_training, prepare_from_root
    from tdub.utils import get_avoids
    from tdub.frames import drop_cols
    qf = quick_files(data_dir)
    extra_sel = extra_selection
    if extra_sel:
        extra_sel = PosixPath(extra_sel).read_text().strip()
    df, y, w = prepare_from_root(
        qf[f"tW_{nlo_method}"],
        qf["ttbar"],
        region,
        weight_mean=1.0,
        extra_selection=extra_sel,
        use_tptrw=use_tptrw,
    )
    drop_cols(df, *get_avoids(region))
    params = dict(
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        max_depth=max_depth,
        n_estimators=n_estimators,
    )
    extra_sum = {"region": region, "nlo_method": nlo_method}
    sr = single_training(
        df,
        y,
        w,
        params,
        out_dir,
        early_stopping_rounds=early_stopping_rounds,
        extra_summary_entries=extra_sum,
    )
    return 0

@cli.command("train-scan", context_settings=dict(max_content_width=140))
@click.argument("config", type=str)
@click.argument("data-dir", type=str)
@click.option("-r", "--region", type=str, required=True, help="the region to train on")
@click.option("-o", "--out-dir", type=str, required=True, help="output directory name")
@click.option("-n", "--nlo-method", type=str, default="DR", help="tW simluation NLO method", show_default=True)
@click.option("-s", "--script-name", type=str, default="hopt.scan.REGION.sub", help="output script name", show_default=True)
@click.option("-x", "--extra-selection", type=str, help="extra selection string")
@click.option("-t", "--use-tptrw", is_flag=True, help="apply top pt reweighting")
@click.option("-e", "--early-stopping-rounds", type=int, help="number of early stopping rounds")
def scan(config, data_dir, region, out_dir, nlo_method, script_name, extra_selection, use_tptrw, early_stopping_rounds):
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
            print(f"Arguments = train-single {run}\nQueue\n\n", file=f)

    return 0


@cli.command("train-check", context_settings=dict(max_content_width=92))
@click.argument("directory", type=str)
@click.option("-p", "--print-top", is_flag=True, help="Print the top results")
@click.option("-n", "--n-res", type=int, default=10, help="Number of top results to print", show_default=True)
def check(directory, print_top, n_res):
    """Check the results of a parameter scan."""
    from tdub.train import SingleTrainingResult
    results = []
    top_dir = PosixPath(directory)
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

    return 0


@cli.command("train-fold", context_settings=dict(max_content_width=92))
@click.option("-s", "--scan-dir", type=str, help="scan step's output directory")
@click.option("-d", "--data-dir", type=str, help="directory containing data files", required=True)
@click.option("-o", "--out-dir", type=str, help="directory to save output", required=True)
@click.option("-t", "--use-tptrw", is_flag=True, help="use top pt reweighting")
@click.option("-r", "--random-seed", type=int, default=414, help="random seed for folding", show_default=True)
@click.option("-n", "--n-splits", type=int, default=3, help="number of splits for folding", show_default=True)
def fold(scan_dir, data_dir, out_dir, use_tptrw, random_seed, n_splits):
    """Perform a folded training based on a hyperparameter scan result"""
    from tdub.train import folded_training, prepare_from_root
    from tdub.frames import drop_cols
    with open(f"{scan_dir}/summary.json", "r") as f:
        summary = json.load(f)
    nlo_method = summary["nlo_method"]
    best_iteration = summary["best_iteration"]
    if best_iteration > 0:
        summary["all_params"]["n_estimators"] = best_iteration
    region = summary["region"]
    branches = summary["features"]
    qf = quick_files(data_dir)
    df, y, w = prepare_from_root(
        qf[f"tW_{nlo_method}"],
        qf["ttbar"],
        region,
        branches=branches,
        weight_mean=1.0,
        use_tptrw=use_tptrw,
    )
    folded_training(
        df,
        y,
        w,
        summary["all_params"],
        {"verbose": 10},
        out_dir,
        summary["region"],
        kfold_kw={
            "n_splits": n_splits,
            "shuffle": True,
            "random_state": random_seed,
        },
    )
    return 0



if __name__ == "__main__":
    import tdub.constants
    tdub.constants.AVOID_IN_CLF_1j1b = []
    tdub.constants.AVOID_IN_CLF_2j1b = []
    tdub.constants.AVOID_IN_CLF_2j2b = []
    cli()
