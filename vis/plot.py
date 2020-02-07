#!/usr/bin/env python3

## stdlib
import os
import pathlib
import argparse

## pip
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pygram11 as pg
import yaml

## tdub
from tdub.utils import (
    quick_files,
    get_branches,
    categorize_branches,
    get_selection,
    bin_centers,
)
from tdub.frames import raw_dataframe


mpl.use("Agg")
mpl.rcParams["figure.figsize"] = (6, 5.25)
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["font.size"] = 12
mpl.rcParams["xtick.top"] = True
mpl.rcParams["ytick.right"] = True
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["ytick.minor.visible"] = True
mpl.rcParams["xtick.major.width"] = 0.8
mpl.rcParams["xtick.minor.width"] = 0.8
mpl.rcParams["xtick.major.size"] = 7.0
mpl.rcParams["xtick.minor.size"] = 4.0
mpl.rcParams["xtick.major.pad"] = 1.5
mpl.rcParams["xtick.minor.pad"] = 1.4
mpl.rcParams["ytick.major.width"] = 0.8
mpl.rcParams["ytick.minor.width"] = 0.8
mpl.rcParams["ytick.major.size"] = 7.0
mpl.rcParams["ytick.minor.size"] = 4.0
mpl.rcParams["ytick.major.pad"] = 1.5
mpl.rcParams["ytick.minor.pad"] = 1.4
mpl.rcParams["legend.frameon"] = False
mpl.rcParams["legend.numpoints"] = 1
mpl.rcParams["legend.fontsize"] = 11
mpl.rcParams["legend.handlelength"] = 1.5
mpl.rcParams["axes.formatter.limits"] = [-4, 4]
mpl.rcParams["axes.formatter.use_mathtext"] = True


DESIRED_SAMPLES = ["tW_DR", "ttbar", "Zjets", "Diboson", "MCNP", "Data"]


with open("meta.yaml") as f:
    META = yaml.load(f, Loader=yaml.Loader)


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="make some plots")
    parser.add_argument("-d", "--data-dir", type=str, default="/Users/ddavis/ATLAS/data/wtloop/v29_20200201", help="data directory")
    parser.add_argument("-o", "--output-dir", type=str, default=".", help="directory to store output")
    parser.add_argument("-r", "--apply-tptrw", action="store_true", help="apply top pt reweighting")
    # fmt: on
    return parser.parse_args()


def ahist(frames, variable, binning, logy, fprefix):
    bins, start, stop = binning
    bedges = np.linspace(start, stop, bins + 1)
    bcenters = bin_centers(bedges)
    counts = {}
    errs = {}
    counts_mc = np.zeros((bins,), dtype=np.float32)
    err_mc = np.zeros((bins,), dtype=np.float32)
    for ds in DESIRED_SAMPLES:
        x = frames[ds][variable].to_numpy()
        w = frames[ds]["weight_nominal"].to_numpy()
        count, err = pg.histogram(x, bins=bins, range=(start, stop), weights=w, flow=True)
        counts[ds] = count
        errs[ds] = err
        if ds != "Data":
            counts_mc += count
            err_mc += err ** 2
    err_mc = np.sqrt(err_mc)
    plot(variable, binning, counts, errs, counts_mc, err_mc, logy, f"{fprefix}_{variable}.pdf")


def plot(variable, binning, counts, errs, counts_mc, err_mc, logy, fname):
    bins, start, stop = binning
    bedges = np.linspace(start, stop, bins + 1)
    bcenters = bin_centers(bedges)
    width = round(bedges[1] - bedges[0], 2)

    ratio = counts["Data"] / counts_mc
    ratio_err = counts["Data"] / (counts_mc ** 2) + np.power(
        counts["Data"] * err_mc / (counts_mc ** 2), 2
    )

    fig, (ax, axr) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(6, 5),
        gridspec_kw=dict(height_ratios=[3.25, 1], hspace=0.025),
    )
    ax.hist(
        [bcenters for _ in range(5)],
        bins=bedges,
        weights=[
            counts["MCNP"],
            counts["Diboson"],
            counts["Zjets"],
            counts["ttbar"],
            counts["tW_DR"],
        ],
        histtype="stepfilled",
        stacked=True,
        label=["MCNP", "Diboson", "$Z$+jets", "$t\\bar{t}$", "$tW$"],
        color=["#9467bd", "#ff7f0e", "#2ca02c", "#d62728", "#1f77b4"],
    )
    ax.errorbar(
        bcenters, counts["Data"], yerr=errs["Data"], label="Data", fmt="ko", zorder=500
    )
    axr.plot([start, stop], [1.0, 1.0], color="gray", linestyle="solid", marker=None)
    axr.errorbar(bcenters, ratio, yerr=ratio_err, fmt="ko", zorder=501)
    axr.set_ylim([0.8, 1.2])
    axr.set_yticks([0.9, 1.0, 1.1])
    xlabel = "{} [{}]".format(
        META["titles"][variable]["mpl"], META["titles"][variable]["unit"]
    )
    xlabel = xlabel.replace(" []", "")
    axr.set_xlabel(xlabel, horizontalalignment="right", x=1.0)
    ylabel = "Events/{} {}".format(width, META["titles"][variable]["unit"])
    ax.set_ylabel(ylabel, horizontalalignment="right", y=1.0)
    axr.set_ylabel("Data/MC")

    ax.legend(loc="upper right")
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0, handles.pop())
    labels.insert(0, labels.pop())
    ax.legend(handles, labels, loc="upper right", ncol=1)
    if logy:
        ax.set_yscale("log")
        ax.set_ylim([10, ax.get_ylim()[1] * 100])
    else:
        ax.set_ylim([0, ax.get_ylim()[1] * 1.35])
    ax.set_xlim([start, stop])
    fig.subplots_adjust(left=0.115, bottom=0.115, right=0.965, top=0.95)
    fig.savefig(fname)
    plt.close(fig)


def main():
    args = parse_args()
    qf = quick_files(args.data_dir)
    branches = get_branches(qf["Data"][0])

    masks1j1b, masks2j1b, masks2j2b = {}, {}, {}
    frames1j1b, frames2j1b, frames2j2b = {}, {}, {}
    print("reading data from disk")
    frames = {
        name: raw_dataframe(qf[name], branches=branches, drop_weight_sys=True)
        for name in DESIRED_SAMPLES
    }
    print("determing selections")
    for ds in DESIRED_SAMPLES:
        if ds != "Data":
            frames[ds]["weight_nominal"] *= 139.0
        if args.apply_tptrw and ds == "ttbar":
            print("applying top pt reweighting")
            frames[ds].apply_weight_tptrw()
        masks1j1b[ds] = frames[ds].eval(get_selection("1j1b"))
        masks2j1b[ds] = frames[ds].eval(get_selection("2j1b"))
        masks2j2b[ds] = frames[ds].eval(get_selection("2j2b"))
        frames1j1b[ds] = frames[ds][masks1j1b[ds]]
        frames2j1b[ds] = frames[ds][masks2j1b[ds]]
        frames2j2b[ds] = frames[ds][masks2j2b[ds]]

    print("frames prepared; starting plotting")

    outdir = pathlib.PosixPath(args.output_dir)
    curdir = pathlib.PosixPath.cwd()
    outdir.mkdir(exist_ok=True)
    os.chdir(outdir)

    for entry in META["regions"]["r1j1b"]:
        binning = (entry["nbins"], entry["xmin"], entry["xmax"])
        ahist(frames1j1b, entry["var"], binning, entry["log"], "1j1b")
    for entry in META["regions"]["r2j1b"]:
        binning = (entry["nbins"], entry["xmin"], entry["xmax"])
        ahist(frames2j1b, entry["var"], binning, entry["log"], "2j1b")
    for entry in META["regions"]["r2j2b"]:
        binning = (entry["nbins"], entry["xmin"], entry["xmax"])
        ahist(frames2j2b, entry["var"], binning, entry["log"], "2j2b")

    os.chdir(curdir)


if __name__ == "__main__":
    main()
