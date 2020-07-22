#!/usr/bin/env python

from tdub.frames import raw_dataframe
from tdub.data import quick_files
from tdub.art import setup_tdub_style, draw_atlas_label
from tdub.hist import bin_centers
import tdub.config

from pygram11 import fix1d
from pathlib import PosixPath
import matplotlib.pyplot as plt
import numpy as np
import click


def drds_comparison(dr_counts, ds_counts, edges):
    differences = dr_counts - ds_counts
    variation = dr_counts + differences
    red_perdiff = (ds_counts - dr_counts) / dr_counts * 100.0
    blu_perdiff = (variation - dr_counts) / dr_counts * 100.0
    full_sample = np.hstack([ds_counts, variation])
    main_max = np.max(full_sample) * 1.6
    full_ratio_sample = np.hstack([red_perdiff, blu_perdiff])
    ratio_min = np.min(full_ratio_sample) * 1.4
    ratio_max = np.max(full_ratio_sample) * 1.4
    fig, (ax, axr) = plt.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw=dict(height_ratios=[3.25, 1], hspace=0.025),
    )
    centers = bin_centers(edges)
    ax.hist(centers, bins=edges, weights=ds_counts, color="red", histtype="step", label=r"$+1\sigma$ (DS variation)")
    ax.hist(centers, bins=edges, weights=variation, color="blue", histtype="step", label=r"$-1\sigma$ (DS symmetrised)")
    ax.hist(centers, bins=edges, weights=dr_counts, color="black", histtype="step", label="Nominal DR")
    ax.set_ylim([0, main_max])
    ax.set_ylabel("Number of Events", horizontalalignment="right", y=1.0)
    ax.legend()
    axr.hist(centers, bins=edges, weights=red_perdiff, color="red", histtype="step")
    axr.hist(centers, bins=edges, weights=blu_perdiff, color="blue", histtype="step")
    axr.set_ylim([ratio_min, ratio_max])
    axr.set_xlim([edges[0], edges[-1]])
    axr.plot(edges, np.zeros_like(edges), ls="-", lw=1.5, c="black")
    axr.set_ylabel(r"$\frac{\mathrm{Sys.} - \mathrm{Nom.}}{\mathrm{Sys.}}$ [%]")
    axr.set_xlabel("BDT Response", horizontalalignment="right", x=1.0)
    fig.subplots_adjust(left=0.15)
    draw_atlas_label(ax, follow="Simulation Internal")
    return fig, ax, axr


@click.group(context_settings=dict(max_content_width=82))
def cli():
    pass


@cli.command("bdt-cut-plots", context_settings=dict(max_content_width=92))
@click.argument("source", type=click.Path(exists=True))
@click.option("--branch", type=str, default="bdtres03", help="BDT branch")
@click.option("--lo-1j1b", type=float, default=0.3, help="Low end 1j1b BDT cut")
@click.option("--hi-2j1b", type=float, default=0.7, help="High end 2j1b BDT cut")
@click.option("--lo-2j2b", type=float, default=0.4, help="Low end 2j2b BDT cut")
@click.option("--hi-2j2b", type=float, default=0.7, help="High end 2j2b BDT cut")
@click.option("--bins-1j1b", type=(int, float, float), default=(20, 0.2, 0.80), help="Binning (n, min, max) of 1j1b bins")
@click.option("--bins-2j1b", type=(int, float, float), default=(20, 0.2, 0.85), help="Binning (n, min, max) of 2j1b bins")
@click.option("--bins-2j2b", type=(int, float, float), default=(20, 0.2, 0.90), help="Binning (n, min, max) of 2j2b bins")
def bdt_cut_plots(
    source,
    branch,
    lo_1j1b,
    hi_2j1b,
    lo_2j2b,
    hi_2j2b,
    bins_1j1b,
    bins_2j1b,
    bins_2j2b,
):
    """Geneate plots showing BDT cuts."""

    setup_tdub_style()
    source = PosixPath(source)
    qf = quick_files(source)

    def drds_histograms(dr_df, ds_df, region, branch="bdtres03", weight_branch="weight_nominal", nbins=12, xmin=0.2, xmax=0.9):
        dr_hist, err = fix1d(dr_df[branch].to_numpy(), bins=nbins, range=(xmin, xmax), weights=dr_df[weight_branch].to_numpy() * 139.0, flow=True)
        ds_hist, err = fix1d(ds_df[branch].to_numpy(), bins=nbins, range=(xmin, xmax), weights=ds_df[weight_branch].to_numpy() * 139.0, flow=True)
        return dr_hist, ds_hist

    branches = [branch, "weight_nominal", "reg1j1b", "reg2j1b", "reg2j2b", "OS"]
    dr_df = raw_dataframe(qf["tW_DR"], branches=branches)
    ds_df = raw_dataframe(qf["tW_DS"], branches=branches)

    ##################

    dr, ds = drds_histograms(
        dr_df.query(tdub.config.SELECTION_1j1b),
        ds_df.query(tdub.config.SELECTION_1j1b),
        "1j1b",
        branch,
        nbins=bins_1j1b[0],
        xmin=bins_1j1b[1],
        xmax=bins_1j1b[2]
    )
    fig, ax, axr = drds_comparison(dr, ds, np.linspace(bins_1j1b[1], bins_1j1b[2], bins_1j1b[0] + 1))
    ax.fill_betweenx([-1, 1.0e5], -1.0, lo_1j1b, color="gray", alpha=0.5)
    axr.fill_betweenx([-200, 200], -1.0, lo_1j1b, color="gray", alpha=0.5)
    fig.savefig("drds_1j1b.pdf")
    plt.close(fig)

    ##################

    dr, ds = drds_histograms(
        dr_df.query(tdub.config.SELECTION_2j1b),
        ds_df.query(tdub.config.SELECTION_2j1b),
        "2j1b",
        branch,
        nbins=bins_2j1b[0],
        xmin=bins_2j1b[1],
        xmax=bins_2j1b[2]
    )
    fig, ax, axr = drds_comparison(dr, ds, np.linspace(bins_2j1b[1], bins_2j1b[2], bins_2j1b[0] + 1))
    ax.fill_betweenx([-1, 1.0e5], hi_2j1b, 1.0, color="gray", alpha=0.5)
    axr.fill_betweenx([-200, 200], hi_2j1b, 1.0, color="gray", alpha=0.5)
    fig.savefig("drds_2j1b.pdf")
    plt.close(fig)

    ##################

    dr, ds = drds_histograms(
        dr_df.query(tdub.config.SELECTION_2j2b),
        ds_df.query(tdub.config.SELECTION_2j2b),
        "2j2b",
        branch,
        nbins=bins_2j2b[0],
        xmin=bins_2j2b[1],
        xmax=bins_2j2b[2]
    )
    fig, ax, axr = drds_comparison(dr, ds, np.linspace(bins_2j2b[1], bins_2j2b[2], bins_2j2b[0] + 1))
    ax.fill_betweenx([-1, 1.0e5], -1.0, lo_2j2b, color="gray", alpha=0.5)
    axr.fill_betweenx([-200, 200], -1.0, lo_2j2b, color="gray", alpha=0.5)
    ax.fill_betweenx([-1, 1.0e5], hi_2j2b, 1.0, color="gray", alpha=0.5)
    axr.fill_betweenx([-200, 200], hi_2j2b, 1.0, color="gray", alpha=0.5)
    fig.savefig("drds_2j2b.pdf")
    plt.close(fig)


def run_cli():
    cli()


if __name__ == "__main__":
    run_cli()
