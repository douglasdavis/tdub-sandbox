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


class Comparison:
    def __init__(self, nominal, up, down, pdiff_up, pdiff_down):
        self.nominal = nominal
        self.up = up
        self.down = down
        self.pdiff_up = pdiff_up
        self.pdiff_down = pdiff_down

    @property
    def pdiff_min(self):
        return np.amin([self.pdiff_up, self.pdiff_down])

    @property
    def pdiff_max(self):
        return np.amax([self.pdiff_up, self.pdiff_down])

    @property
    def template_max(self):
        return np.amax([self.up, self.down])


def onesidedsym_comparison(nominal, up):
    """Generate components of a systematic comparion plot.

    Paramters
    ---------
    nominal : numpy.ndarray
        Histogram bin counts for the nominal template.
    up : numpy.ndarray
        Histogram bin counts for the "up" variation.

    Returns
    -------
    Comparison
        The complete description of the comparison
    """
    diffs = nominal - up
    down = nominal + diffs
    pdiff_up = (up - nominal) / nominal * 100.0
    pdiff_down = (down - nominal) / nominal * 100.0
    return Comparison(nominal, up, down, pdiff_up, pdiff_down)


def drds_comparison(dr_counts, ds_counts, edges):
    c = onesidedsym_comparison(dr_counts, ds_counts)
    centers = bin_centers(edges)
    fig, (ax, axr) = plt.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[3.25, 1], hspace=0.025))

    ax.hist(centers, bins=edges, weights=c.up, color="red", histtype="step", label=r"$+1\sigma$ Variation")
    ax.hist(centers, bins=edges, weights=c.down, color="blue", histtype="step", label=r"$-1\sigma$ Variation")
    ax.hist(centers, bins=edges, weights=c.nominal, color="black", histtype="step", label="Nominal")
    ymax = c.template_max * 1.6
    ax.set_ylim([0, ymax])
    ax.set_ylabel("Number of Events", horizontalalignment="right", y=1.0)
    ax.legend()

    axr.hist(centers, bins=edges, weights=c.pdiff_up, color="red", histtype="step")
    axr.hist(centers, bins=edges, weights=c.pdiff_down, color="blue", histtype="step")
    axr.set_ylim([c.pdiff_min * 1.25, c.pdiff_max * 1.25])
    axr.set_xlim([edges[0], edges[-1]])
    axr.plot(edges, np.zeros_like(edges), ls="-", lw=1.5, c="black")
    axr.set_ylabel(r"$\frac{\mathrm{Sys.} - \mathrm{Nom.}}{\mathrm{Sys.}}$ [%]")
    axr.set_xlabel("BDT Response", horizontalalignment="right", x=1.0)

    fig.subplots_adjust(left=0.15)
    draw_atlas_label(ax, follow="Simulation Internal", follow_shift=0.17)
    return fig, ax, axr


@click.command("bdt-cut-plots")
@click.argument("source", type=click.Path(exists=True))
@click.option("--branch", type=str, default="bdtres03", help="BDT branch")
@click.option("--lo-1j1b", type=float, default=0.35, help="Low end 1j1b BDT cut")
@click.option("--hi-2j1b", type=float, default=0.70, help="High end 2j1b BDT cut")
@click.option("--lo-2j2b", type=float, default=0.45, help="Low end 2j2b BDT cut")
@click.option("--hi-2j2b", type=float, default=0.775, help="High end 2j2b BDT cut")
@click.option("--bins-1j1b", type=(int, float, float), default=(18, 0.2, 0.75), help="Binning (n, min, max) of 1j1b bins")
@click.option("--bins-2j1b", type=(int, float, float), default=(18, 0.2, 0.85), help="Binning (n, min, max) of 2j1b bins")
@click.option("--bins-2j2b", type=(int, float, float), default=(18, 0.2, 0.90), help="Binning (n, min, max) of 2j2b bins")
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
    ymid = ax.get_ylim()[1] * 0.69
    xmid = (lo_1j1b - ax.get_xlim()[0]) * 0.5 + ax.get_xlim()[0]
    ax.text(xmid, ymid, "Excluded", ha="center", va="center", color="gray", size=9)
    ax.fill_betweenx([-1, 1.0e5], -1.0, lo_1j1b, color="gray", alpha=0.55)
    axr.fill_betweenx([-200, 200], -1.0, lo_1j1b, color="gray", alpha=0.55)
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
    ax.fill_betweenx([-1, 1.0e5], hi_2j1b, 1.0, color="gray", alpha=0.55)
    axr.fill_betweenx([-200, 200], hi_2j1b, 1.0, color="gray", alpha=0.55)
    ymid = ax.get_ylim()[1] * 0.69
    xmid = (ax.get_xlim()[1] - hi_2j1b) * 0.5 + hi_2j1b
    ax.text(xmid, ymid, "Excluded", ha="center", va="center", color="gray", size=9)
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
    ax.fill_betweenx([-1, 1.0e5], -1.0, lo_2j2b, color="gray", alpha=0.55)
    axr.fill_betweenx([-200, 200], -1.0, lo_2j2b, color="gray", alpha=0.55)
    ax.fill_betweenx([-1, 1.0e5], hi_2j2b, 1.0, color="gray", alpha=0.55)
    axr.fill_betweenx([-200, 200], hi_2j2b, 1.0, color="gray", alpha=0.55)
    ymid = ax.get_ylim()[1] * 0.69
    xmid = (lo_2j2b - ax.get_xlim()[0]) * 0.5 + ax.get_xlim()[0]
    ax.text(xmid, ymid, "Excluded", ha="center", va="center", color="gray", size=9)
    xmid = (ax.get_xlim()[1] - hi_2j2b) * 0.5 + hi_2j2b
    ax.text(xmid, ymid, "Excluded", ha="center", va="center", color="gray", size=9)
    fig.savefig("drds_2j2b.pdf")
    plt.close(fig)


if __name__ == "__main__":
    bdt_cut_plots()
