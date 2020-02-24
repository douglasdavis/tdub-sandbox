#!/usr/bin/env python

import argparse
import logging
from typing import Dict, Any, Union, Optional

from tdub import setup_logging
setup_logging()
log = logging.getLogger("tdub-sandbox.xgb")
from tdub.frames import drop_cols
from tdub.train import (
    prepare_from_root,
    single_training,
)
from tdub.utils import quick_files, get_avoids, get_features, augment_features
import tdub.constants


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
    )
    drop_cols(df, *get_avoids(args.region))
    params = dict(
        learning_rate=args.learning_rate,
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
        use_xgboost=True,
    )
    return sr


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="woooo")
    parser.add_argument("-d", "--data-dir", type=str, help="directory containing data files", required=True)
    parser.add_argument("-r", "--region", type=str, help="analysis region", required=True)
    parser.add_argument("-n", "--nlo-method", type=str, choices=["DR", "DS"], default="DR", help="NLO method")
    parser.add_argument("-o", "--out-dir", type=str, help="output directory", required=True)
    parser.add_argument("-e", "--extra-selection", type=str, help="input file listing extra selections before training")
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--max-depth", type=int, required=True)
    parser.add_argument("--n-estimators", type=int, required=True)
    parser.add_argument("--early-stopping-rounds", type=int, help="early stopping rounds")
    return parser.parse_args()


def main():
    args = get_args()
    single(args)


if __name__ == "__main__":
    main()
