#!/usr/bin/env python3

## stdlib
import os
import pathlib
import logging
from typing import Dict, Tuple, List, Optional, Any

## pip
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygram11 as pg
import yaml
import click

## tdub
from tdub.constants import FEATURESET_1j1b, FEATURESET_2j1b, FEATURESET_2j2b
from tdub.art import setup_style, canvas_from_counts
from tdub.utils import (
    quick_files,
    get_branches,
    categorize_branches,
    get_selection,
    bin_centers,
)
from tdub.frames import raw_dataframe, apply_weight_tptrw
from tdub import setup_logging

setup_style()
setup_logging()

log = logging.getLogger("plot.py")


META_FILE = open("meta.yml", "r")
ALL_SAMPLES = ["tW_DR", "ttbar", "Zjets", "Diboson", "MCNP", "Data"]
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


def save_and_close(fig: plt.Figure, name: str) -> None:
    """save a figure and close it

    Parameters
    ----------
    fig : :obj:`matplotlib.figure.Figure`
        the matplotlib figure to save
    name : str
        the filename to give the saved figure
    """
    log.info("saving %s" % name)
    fig.savefig(name)
    plt.close(fig)


def tune_axes(
    ax: plt.Axes,
    axr: plt.Axes,
    variable: str,
    binning: Tuple[int, float, float],
    logy: bool = False,
    linscale: float = 1.4,
    logscale: float = 100,
) -> None:
    """tune up the axes properties

    Parameters
    ----------
    ax : :obj:`matplotlib.axes.Axes`
        the main stack axes
    axr : :obj:`matplotlib.axes.Axes`
        the ratio axes
    variable : str
        the name for the variable that is histogrammed
    binning : tuple(int, float, float)
        the number of bins and the start and stop on the x-axis
    logy : bool
        set the yscale to log
    linscale : float
        the factor to scale up the y-axis when linear
    logscale : float
        the factor to scale up the y-axis when log
    """
    nbins, start, stop = binning
    width = round((stop - start) / nbins, 2)
    set_labels(ax, axr, variable, width=width)
    ax.set_xlim([start, stop])
    axr.set_xlim([start, stop])
    if logy:
        ax.set_yscale("log")
        ax.set_ylim([10, ax.get_ylim()[1] * logscale])
    else:
        ax.set_ylim([0, ax.get_ylim()[1] * linscale])


def plot_from_region_frames(
    frames: Dict[str, pd.DataFrame],
    variable: str,
    binning: Tuple[int, float, float],
    region_label: str,
    logy: bool = False,
    legend_kw : Dict[str, Any] = None,
) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
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
    legend_kw : dict(str, Any)
        keyward arguments passed to :py:func:`matplotlib.Axes.axes.legend`.

    """
    if variable not in frames["Data"].columns.to_list():
        log.warning("%s not in dataframe; skipping" % variable)
        return None, None, None
    nbins, start, stop = binning
    bin_edges = np.linspace(start, stop, nbins + 1)
    counts = {}
    errors = {}
    for samp in ALL_SAMPLES:
        x = frames[samp][variable].to_numpy()
        w = frames[samp]["weight_nominal"].to_numpy()
        count, err = pg.histogram(x, bins=nbins, range=(start, stop), weights=w, flow=True)
        counts[samp] = count
        errors[samp] = err
    fig, ax, axr = canvas_from_counts(counts, errors, bin_edges)

    draw_atlas_label(
        ax, extra_lines=["$\sqrt{s} = 13$ TeV, $L = 139$ fb$^{-1}$", region_label]
    )
    tune_axes(ax, axr, variable, binning, logy=logy)

    if legend_kw is None:
        legend_kw = {}
    legend_kw["ncol"] = 2
    ax.legend(loc="upper right")
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0, handles.pop())
    labels.insert(0, labels.pop())
    ax.legend(handles, labels, loc="upper right", **legend_kw)

    fig.subplots_adjust(left=0.125, bottom=0.095, right=0.965, top=0.95)
    return fig, ax, axr


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
        for name in ALL_SAMPLES
    }
    log.info("determing selections")
    for samp in ALL_SAMPLES:
        if samp != "Data":
            frames[samp]["weight_nominal"] *= LUMI
        if apply_tptrw and samp == "ttbar":
            log.info("applying top pt reweighting")
            frames[samp].apply_weight_tptrw()
        masks1j1b[samp] = frames[samp].eval(get_selection("1j1b"))
        masks2j1b[samp] = frames[samp].eval(get_selection("2j1b"))
        masks2j2b[samp] = frames[samp].eval(get_selection("2j2b"))
        frames1j1b[samp] = frames[samp][masks1j1b[samp]]
        frames2j1b[samp] = frames[samp][masks2j1b[samp]]
        frames2j2b[samp] = frames[samp][masks2j2b[samp]]

    return frames1j1b, frames2j1b, frames2j2b


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################




@click.command(context_settings=dict(max_content_width=92))
@click.option("-d", "--data-dir", type=str, default="/Users/ddavis/ATLAS/data/wtloop/v29_20200201", help="data directory")
@click.option("-o", "--output-dir", type=str, default="pdfs", help="directory to store output")
@click.option("--apply-tptrw", is_flag=True, help="apply top pt reweighting")
@click.option("--from-parquet", is_flag=True, help="use parquet files")
@click.option("--prep-parquet", is_flag=True, help="data prep to parquet")
@click.option("--regions", type=str, multiple=True, help="regions to plot", default=["2j2b", "2j1b", "1j1b"])
@click.option("--skip-absent-features", is_flag=True, help="skip if var not in feature list")
def plot(data_dir, output_dir, apply_tptrw, from_parquet, prep_parquet, regions, skip_absent_features):
    curdir = pathlib.PosixPath(__file__).parent.resolve()
    datadir = curdir / "data"
    datadir.mkdir(exist_ok=True)

    if from_parquet:
        log.info("reading parquet files")
        dfs_1j1b, dfs_2j1b, dfs_2j2b = {}, {}, {}
        for samp in ALL_SAMPLES:
            dfs_1j1b[samp] = pd.read_parquet(datadir / f"{samp}_1j1b.parquet")
            dfs_2j1b[samp] = pd.read_parquet(datadir / f"{samp}_2j1b.parquet")
            dfs_2j2b[samp] = pd.read_parquet(datadir / f"{samp}_2j2b.parquet")
        log.info("done reading parquet files")

    else:
        qf = quick_files(data_dir)
        dfs_1j1b, dfs_2j1b, dfs_2j2b = region_frames_from_qf(qf)
        if prep_parquet:
            log.info("preping parquet files")
            for k, v in dfs_1j1b.items():
                name = datadir / f"{k}_1j1b.parquet"
                v.to_parquet(name)
            for k, v in dfs_2j1b.items():
                name = datadir / f"{k}_2j1b.parquet"
                v.to_parquet(name)
            for k, v in dfs_2j2b.items():
                name = datadir / f"{k}_2j2b.parquet"
                v.to_parquet(name)
            log.info("dont prepping parquet")
            exit(0)

    if apply_tptrw:
        log.info("applying top pt reweighting")
        apply_weight_tptrw(dfs_1j1b["ttbar"])
        apply_weight_tptrw(dfs_2j1b["ttbar"])
        apply_weight_tptrw(dfs_2j2b["ttbar"])

    plotdir = pathlib.PosixPath(output_dir)
    plotdir.mkdir(exist_ok=True)
    os.chdir(plotdir)

    if "1j1b" in regions:
        for entry in META["regions"]["r1j1b"]:
            if skip_absent_features:
                if entry["var"] not in FEATURESET_1j1b:
                    continue
            binning = (entry["nbins"], entry["xmin"], entry["xmax"])
            fig, ax, axr = plot_from_region_frames(
                dfs_1j1b, entry["var"], binning, "1j1b", entry["log"]
            )
            if fig is not None:
                save_and_close(fig, "r{}_{}.pdf".format("1j1b", entry["var"]))
    if "2j1b" in regions:
        for entry in META["regions"]["r2j1b"]:
            if skip_absent_features:
                if entry["var"] not in FEATURESET_2j1b:
                    continue
            binning = (entry["nbins"], entry["xmin"], entry["xmax"])
            fig, ax, axr = plot_from_region_frames(
                dfs_2j1b, entry["var"], binning, "2j1b", entry["log"]
            )
            if fig is not None:
                save_and_close(fig, "r{}_{}.pdf".format("2j1b", entry["var"]))
    if "2j2b" in regions:
        for entry in META["regions"]["r2j2b"]:
            if skip_absent_features:
                if entry["var"] not in FEATURESET_2j2b:
                    continue
            binning = (entry["nbins"], entry["xmin"], entry["xmax"])
            fig, ax, axr = plot_from_region_frames(
                dfs_2j2b, entry["var"], binning, "2j2b", entry["log"]
            )
            if fig is not None:
                save_and_close(fig, "r{}_{}.pdf".format("2j2b", entry["var"]))

    os.chdir(curdir)


if __name__ == "__main__":
    plot()
