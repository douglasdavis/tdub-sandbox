#!/usr/bin/env python3

## stdlib
import os
import pathlib
import argparse
import logging

from typing import Dict, Tuple, List, Optional, Any

## pip
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from tdub import setup_logging

setup_logging()

log = logging.getLogger("plot.py")


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


META_FILE = open("meta.yaml", "r")
DESIRED_SAMPLES = ["tW_DR", "ttbar", "Zjets", "Diboson", "MCNP", "Data"]
LUMI = 139.0
META = yaml.load(META_FILE, Loader=yaml.Loader)
META_FILE.close()


def draw_atlas_label(
    ax: plt.Axes,
    internal: bool = True,
    extra_lines: Optional[Any] = None,
    x: float = 0.050,
    y: float = 0.905,
    s1: int = 14,
    s2: int = 12,
) -> None:
    ax.text(
        x,
        y,
        "ATLAS",
        fontstyle="italic",
        fontweight="bold",
        transform=ax.transAxes,
        size=s1,
    )
    if internal:
        ax.text(x + 0.15, y, r"Internal", transform=ax.transAxes, size=s1)
    if extra_lines is not None:
        for i, exline in enumerate(extra_lines):
            ax.text(x, y - (i + 1) * 0.06, exline, transform=ax.transAxes, size=s2)


def set_labels(ax: plt.Axes, axr: plt.Axes, variable: str, width: float) -> None:
    """set the labels on axes for a given variable and bin width

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        the main stack axes
    axr : matplotlib.axes.Axes
        the ratio axes
    variable : str
        the name of the variable being histogrammed
    width : float
        the bin width
    """
    xlabel = "{} [{}]".format(
        META["titles"][variable]["mpl"], META["titles"][variable]["unit"]
    )
    xlabel = xlabel.replace(" []", "")
    axr.set_xlabel(xlabel, horizontalalignment="right", x=1.0)
    ylabel = "Events/{} {}".format(width, META["titles"][variable]["unit"])
    ax.set_ylabel(ylabel, horizontalalignment="right", y=1.0)
    axr.set_ylabel("Data/MC")


def fig_from_counts(
    counts: Dict[str, np.ndarray], errors: Dict[str, np.ndarray], bin_edges: np.ndarray,
) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
    """create a histogram plot given a set of counts and bin edges

    Parameters
    ----------
    counts : dict(str, np.ndarray)
        a dictionary pairing samples to bin counts
    errors : dict(str, np.ndarray)
        a dictionray pairing samples to bin count errors
    bin_edges : np.ndarray
        the histogram bin edges

    Returns
    -------
    fig : matplotlib.figure.Figure
        the matplotlib figure
    ax : matplotlib.axes.Axes
        the matplotlib axes for the histogram stack
    axr : matplotlib.axes.Axes
        the matplotlib axes for the ratio comparison
    """
    centers = bin_centers(bin_edges)
    start, stop = bin_edges[0], bin_edges[-1]
    mc_counts = np.zeros_like(centers, dtype=np.float32)
    mc_errs = np.zeros_like(centers, dtype=np.float32)
    for key in counts.keys():
        if key != "Data":
            mc_counts += counts[key]
            mc_errs += errors[key] ** 2
    mc_errs = np.sqrt(mc_errs)
    ratio = counts["Data"] / mc_counts
    ratio_err = counts["Data"] / (mc_counts ** 2) + np.power(
        counts["Data"] * mc_errs / (mc_counts ** 2), 2
    )
    fig, (ax, axr) = plt.subplots(
        2, 1, sharex=True, gridspec_kw=dict(height_ratios=[3.25, 1], hspace=0.025),
    )
    ax.hist(
        [centers for _ in range(5)],
        bins=bin_edges,
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
        centers, counts["Data"], yerr=errors["Data"], label="Data", fmt="ko", zorder=500
    )
    axr.plot([start, stop], [1.0, 1.0], color="gray", linestyle="solid", marker=None)
    axr.errorbar(centers, ratio, yerr=ratio_err, fmt="ko", zorder=501)
    axr.set_ylim([0.75, 1.25])
    axr.set_yticks([0.8, 0.9, 1.0, 1.1, 1.2])

    return fig, ax, axr


def fig_from_frames(frames, variable, binning) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
    """create a histogram plot from dataframes

    Parameters
    ----------
    frames : dict(str, pd.DataFrame)
        the dataframes for all samples
    variable : str
        the variable we want to histogram
    binning : tuple(int, float, float)
        the bin definition

    Returns
    -------
    fig : matplotlib.figure.Figure
        the matplotlib figure
    ax : matplotlib.axes.Axes
        the matplotlib axes for the histogram stack
    axr : matplotlib.axes.Axes
        the matplotlib axes for the ratio comparison

    """
    nbins, start, stop = binning
    bin_edges = np.linspace(start, stop, nbins + 1)
    counts = {}
    errors = {}
    for ds in DESIRED_SAMPLES:
        x = frames[ds][variable].to_numpy()
        w = frames[ds]["weight_nominal"].to_numpy()
        count, err = pg.histogram(x, bins=nbins, range=(start, stop), weights=w, flow=True)
        counts[ds] = count
        errors[ds] = err

    return fig_from_counts(counts, errors, bin_edges)


def plot_from_region_frames(frames, variable, binning, region_label, logy=False) -> None:
    """create a histogram plot pdf from dataframes and a desired variable

    Parameters
    ----------
    frames : dict(str, pd.DataFrame)
        the dataframes for all samples
    variable : str
        the variable we want to histogram
    binning : tuple(int, float, float)
        the bin definition
    region_label : str
        the region label (will be part of out file name)
    logy : bool
        if true set the yscale to log

    """
    fig, ax, axr = fig_from_frames(frames, variable, binning)
    nbins, start, stop = binning
    width = round((stop - start) / nbins, 2)
    set_labels(ax, axr, variable, width=width)
    draw_atlas_label(ax, extra_lines=["$\sqrt{s} = 13$ TeV, $L = 139$ fb$^{-1}$", region_label])
    ax.set_xlim([start, stop])
    axr.set_xlim([start, stop])
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
    fig.subplots_adjust(left=0.115, bottom=0.115, right=0.965, top=0.95)
    fname = f"{region_label}_{variable}.pdf"
    fig.savefig(fname)
    plt.close(fig)


def region_frames_from_qf(
    qf_result: Dict[str, List[str]], apply_tptrw: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """get dataframes for our desired samples

    Parameters
    ----------
    qf_result : dict(str, list(str))
        the dictionary from calling quick_files
    apply_tptrw : bool
        if True apply tptrw to the ttbar frames

    Returns
    -------
    frames1j1b : dict(str, pd.DataFrame)
        the 1j1b dataframes
    frames2j1b : dict(str, pd.DataFrame)
        the 1j1b dataframes
    frames2j2b : dict(str, pd.DataFrame)
        the 2j2b dataframes
    """
    branches = get_branches(qf_result["Data"][0])
    masks1j1b, masks2j1b, masks2j2b = {}, {}, {}
    frames1j1b, frames2j1b, frames2j2b = {}, {}, {}
    log.info("reading data from disk")
    frames = {
        name: raw_dataframe(qf_result[name], branches=branches, drop_weight_sys=True)
        for name in DESIRED_SAMPLES
    }
    log.info("determing selections")
    for ds in DESIRED_SAMPLES:
        if ds != "Data":
            frames[ds]["weight_nominal"] *= LUMI
        if apply_tptrw and ds == "ttbar":
            log.info("applying top pt reweighting")
            frames[ds].apply_weight_tptrw()
        masks1j1b[ds] = frames[ds].eval(get_selection("1j1b"))
        masks2j1b[ds] = frames[ds].eval(get_selection("2j1b"))
        masks2j2b[ds] = frames[ds].eval(get_selection("2j2b"))
        frames1j1b[ds] = frames[ds][masks1j1b[ds]]
        frames2j1b[ds] = frames[ds][masks2j1b[ds]]
        frames2j2b[ds] = frames[ds][masks2j2b[ds]]

    return frames1j1b, frames2j1b, frames2j2b


##################
##################
##################
##################
##################
##################


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="make some plots")
    parser.add_argument("-d", "--data-dir", type=str,
                        default="/Users/ddavis/ATLAS/data/wtloop/v29_20200201", help="data directory")
    parser.add_argument("-o", "--output-dir", type=str, default=".", help="directory to store output")
    parser.add_argument("-r", "--apply-tptrw", action="store_true", help="apply top pt reweighting")
    # fmt: on
    return parser.parse_args()


def main():
    args = parse_args()
    qf = quick_files(args.data_dir)
    dfs_1j1b, dfs_2j1b, dfs_2j2b = region_frames_from_qf(qf)
    log.info("frames prepared; starting plotting")
    outdir = pathlib.PosixPath(args.output_dir)
    curdir = pathlib.PosixPath.cwd()
    outdir.mkdir(exist_ok=True)
    os.chdir(outdir)

    for entry in META["regions"]["r1j1b"]:
        binning = (entry["nbins"], entry["xmin"], entry["xmax"])
        plot_from_region_frames(dfs_1j1b, entry["var"], binning, "1j1b", entry["log"])
    for entry in META["regions"]["r2j1b"]:
        binning = (entry["nbins"], entry["xmin"], entry["xmax"])
        plot_from_region_frames(dfs_2j1b, entry["var"], binning, "2j1b", entry["log"])
    for entry in META["regions"]["r2j2b"]:
        binning = (entry["nbins"], entry["xmin"], entry["xmax"])
        plot_from_region_frames(dfs_2j2b, entry["var"], binning, "2j2b", entry["log"])

    os.chdir(curdir)


if __name__ == "__main__":
    main()
