#!/usr/bin/env python

# stdlib
from pprint import pprint
import argparse
import sys

# ext
import numpy as np
import pygram11

# tdub
from tdub.frames import iterative_selection
from tdub.data import quick_files


def main(datadir: str, sample: str = "tW_DR"):
    qf = quick_files(datadir)

    files_PP8 = qf[f"{sample}_AFII"]
    if sample == "ttbar":
        files_PH7 = qf[f"{sample}_PS713"]
    else:
        files_PH7 = qf[f"{sample}_PS"]

    print("PP8 files:")
    pprint(files_PP8)
    print("PH7 files:")
    pprint(files_PH7)

    df_PP8 = iterative_selection(
        files_PP8,
        "(OS == True)",
        branches=["OS", "reg1j1b", "reg2j1b", "reg2j2b"],
    )
    df_PH7 = iterative_selection(
        files_PH7,
        "(OS == True)",
        branches=["OS", "reg1j1b", "reg2j1b", "reg2j2b"],
    )
    PP8_raw_sum = df_PP8.weight_nominal.sum()
    PH7_raw_sum = df_PH7.weight_nominal.sum()

    overall_norm_unc = abs(PH7_raw_sum - PP8_raw_sum) / PP8_raw_sum
    print("overall_norm_unc:", overall_norm_unc)

    scale_fac_for_PH7 = PP8_raw_sum / PH7_raw_sum
    print("scale_fac:", scale_fac_for_PH7)

    PP8_1j1b = df_PP8.query("reg1j1b==True").weight_nominal.sum()
    PP8_2j1b = df_PP8.query("reg2j1b==True").weight_nominal.sum()
    PP8_2j2b = df_PP8.query("reg2j2b==True").weight_nominal.sum()

    PH7_1j1b = df_PH7.query("reg1j1b==True").weight_nominal.sum() * scale_fac_for_PH7
    PH7_2j1b = df_PH7.query("reg2j1b==True").weight_nominal.sum() * scale_fac_for_PH7
    PH7_2j2b = df_PH7.query("reg2j2b==True").weight_nominal.sum() * scale_fac_for_PH7

    mig_1j1b = abs(PH7_1j1b - PP8_1j1b) / PP8_1j1b
    mig_2j1b = abs(PH7_2j1b - PP8_2j1b) / PP8_2j1b
    mig_2j2b = abs(PH7_2j2b - PP8_2j2b) / PP8_2j2b

    print("mig_1j1b:", mig_1j1b)
    print("mig_2j1b:", mig_2j1b)
    print("mig_2j2b:", mig_2j2b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-dir", type=str, help="directory with data", required=True)
    parser.add_argument("-s", "--sample", type=str, choices=["ttbar", "tW_DR"], help="sample name", required=True)
    args = parser.parse_args()
    main(args.data_dir, args.sample)
