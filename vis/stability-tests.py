#!/usr/bin/env python

import os
from pathlib import PosixPath
from typing import Union

import click
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np

import tdub.rex as tr


def excluded_summary(
    rex_dir: Union[str, os.PathLike],
    poi: str = "SigXsecOverSM",
):
    rex_dir = PosixPath(rex_dir)
    fit_dir = rex_dir / "Fits"
    fit_name = str(rex_dir.name)

    nominal_result = tr.fit_parameter(fit_dir / f"{fit_name}.txt", name=poi)

    fits = []
    for f in fit_dir.glob(f"{fit_name}_exclude-*.txt"):
        par_name = f.stem.split(f"{fit_name}_exclude-")[-1]
        if "saturatedModel" in par_name:
            continue
        fits.append(par_name)
    fits = sorted(fits, key=str.lower)

    tests = {}
    for pn in fits:
        par_file = rex_dir / "Fits" / f"{fit_name}_exclude-{pn}.txt"
        res = tr.fit_parameter(par_file, name="SigXsecOverSM")
        res.label = tr.prettify_label(pn)
        tests[pn] = res

    names, labels, vals = [], [], []
    for name, res in tests.items():
        names.append(name)
        labels.append(res.label)
        vals.append(tr.delta_param(nominal_result, res))

    vals = np.array(
        vals,
        dtype=[
            ("c", np.float64),
            ("u", np.float64),
            ("d", np.float64)
        ]
    )

    return nominal_result, names, labels, vals


def camp_stab_test(umbrella: PosixPath, fit_name: str = "tW"):
    nominal = umbrella / f"main.force-data.d/{fit_name}/Fits/{fit_name}.txt"
    only_a = umbrella / f"main_only1516.force-data.d/{fit_name}/Fits/{fit_name}.txt"
    only_d = umbrella / f"main_only17.force-data.d/{fit_name}/Fits/{fit_name}.txt"
    only_e = umbrella / f"main_only18.force-data.d/{fit_name}/Fits/{fit_name}.txt"
    fit_n = tr.fit_parameter(nominal, name="SigXsecOverSM")
    fit_a = tr.fit_parameter(only_a, name="SigXsecOverSM")
    fit_d = tr.fit_parameter(only_d, name="SigXsecOverSM")
    fit_e = tr.fit_parameter(only_e, name="SigXsecOverSM")

    labels = ["2015/2016", "2017", "2018"]
    deltas = [tr.delta_param(fit_n, f) for f in (fit_a, fit_d, fit_e)]
    vals = np.array(
        deltas,
        dtype=[
            ("c", np.float64),
            ("u", np.float64),
            ("d", np.float64)
        ]
    )

    return fit_n, labels, labels, vals


def reg_stab_test(umbrella: PosixPath, fit_name: str = "tW"):
    nominal = umbrella / f"main.force-data.d/{fit_name}/Fits/{fit_name}.txt"
    only_1j1b = umbrella / f"main_1j1b.force-data.d/{fit_name}/Fits/{fit_name}.txt"
    only_1j1b2j1b = umbrella / f"main_1j1b2j1b.force-data.d/{fit_name}/Fits/{fit_name}.txt"
    only_1j1b2j2b = umbrella / f"main_1j1b2j2b.force-data.d/{fit_name}/Fits/{fit_name}.txt"
    fit_n = tr.fit_parameter(nominal, name="SigXsecOverSM")
    fit_1j1b = tr.fit_parameter(only_1j1b, name="SigXsecOverSM")
    fit_1j1b2j1b = tr.fit_parameter(only_1j1b2j1b, name="SigXsecOverSM")
    fit_1j1b2j2b = tr.fit_parameter(only_1j1b2j2b, name="SigXsecOverSM")

    labels = ["1j1b only", "1j1b + 2j1b", "1j1b + 2j2b"]
    deltas = [tr.delta_param(fit_n, f) for f in (fit_1j1b, fit_1j1b2j1b, fit_1j1b2j2b)]
    vals = np.array(
        deltas,
        dtype=[
            ("c", np.float64),
            ("u", np.float64),
            ("d", np.float64)
        ]
    )

    return fit_n, labels, labels, vals


def b0_by_year(umbrella: PosixPath, fit_name: str = "tW"):
    nominal = umbrella / f"main.force-data.d/{fit_name}/Fits/{fit_name}.txt"
    only_a = umbrella / f"main_only1516.force-data.d/{fit_name}/Fits/{fit_name}.txt"
    only_d = umbrella / f"main_only17.force-data.d/{fit_name}/Fits/{fit_name}.txt"
    only_e = umbrella / f"main_only18.force-data.d/{fit_name}/Fits/{fit_name}.txt"
    fit_n = tr.fit_parameter(nominal, name="B_ev_B_0")
    fit_a = tr.fit_parameter(only_a, name="B_ev_B_0")
    fit_d = tr.fit_parameter(only_d, name="B_ev_B_0")
    fit_e = tr.fit_parameter(only_e, name="B_ev_B_0")
    vals = np.array(
        [(v.central, v.sig_hi, v.sig_lo) for v in [fit_e, fit_d, fit_a, fit_n]],
        dtype=[
            ("c", np.float64),
            ("u", np.float64),
            ("d", np.float64),
        ]
    )
    ylabs = ["2015/2016", "2017", "2018", "Complete"]
    yvals = np.arange(1, len(ylabs) + 1)
    fig, ax = plt.subplots(figsize=(5.2, 1.5 + len(ylabs) * 0.315))
    ax.set_title(r"Zeroth $b$-tagging B eigenvector NP")
    ax.set_xlim([-2.25, 2.25])
    ax.fill_betweenx([-50, 500], -2, 2, color="yellow", alpha=0.8)
    ax.fill_betweenx([-50, 500], -1, 1, color="green", alpha=0.8)
    ax.set_xlabel(r"$(\hat{\theta} - \theta_0)/\Delta\theta$")
    ax.set_yticks(yvals)
    ax.set_yticklabels(ylabs)
    ax.set_ylim([0.0, len(yvals) + 1])
    ax.errorbar(vals["c"], yvals, xerr=[abs(vals["d"]), vals["u"]], fmt="ko", lw=2, elinewidth=2.25, capsize=3.5)
    for xv, yv in zip(vals["c"], yvals):
        t = f"{xv:1.3f}"
        ax.text(xv, yv + 0.075, t, ha="center", va="bottom", size=10)
    ax.grid(color="black", alpha=0.15)
    #fig.subplots_adjust(left=0.20, right=0.95, bottom=0.2, top=0.8)

    return fig, ax


def make_delta_mu_plot(ax, nom_down, nom_up, xvals, xerlo, xerhi, ylabs):
    yvals = np.arange(1, len(xvals) + 1)
    ax.fill_betweenx(
        [-50, 500],
        nom_down,
        nom_up,
        color="gray",
        alpha=0.5,
        label="Nominal Fit Uncertainty",
    )
    ax.set_xlabel(r"$\Delta\mu=\mu_{tW}^{\mathrm{nominal}}-\mu_{tW}^{\mathrm{test}}$")
    for xv, yv in zip(xvals, yvals):
        t = f"{xv:1.3f}"
        ax.text(xv, yv + 0.075, t, ha="center", va="bottom", size=10)
    ax.set_yticks(yvals)
    ax.set_yticklabels(ylabs)
    ax.set_ylim([0.0, len(yvals) + 1])
    ax.errorbar(
        xvals,
        yvals,
        xerr=[abs(xerlo), xerhi],
        label="Individual tests",
        fmt="ko",
        lw=2,
        elinewidth=2.25,
        capsize=3.5,
    )
    ax.grid(color="black", alpha=0.15)
    ax.legend(bbox_to_anchor=(-1, 0.97, 0, 0), loc="lower left")
    return ax


@click.command()
@click.option("-u", "--umbrella-dir", type=click.Path(exists=True), default=None, help="umbrella directory")
def main(umbrella_dir):
    """Run stability tests via an umbrella fit directory."""
    if umbrella_dir is None:
        umbrella = PosixPath("/opt/spar/analysis/run/fitting/202009/03/core/704/704")
    else:
        umbrella = PosixPath(umbrella_dir)

    nom, names, labels, vals = excluded_summary(umbrella / "main.force-data.d" / "tW")
    fig, ax = plt.subplots(figsize=(5.2, 1.5 + len(names) * 0.315))
    fig.subplots_adjust(left=0.50, right=0.925)
    make_delta_mu_plot(ax, nom.sig_hi, nom.sig_lo, vals["c"], vals["d"], vals["u"], labels)
    fig.savefig("stability-tests-sys-drops.pdf")

    nom, names, labels, vals = camp_stab_test(umbrella)
    fig, ax = plt.subplots(figsize=(5.2, 1.5 + len(names) * 0.315))
    fig.subplots_adjust(left=0.350, right=0.925, bottom=0.3, top=0.99)
    make_delta_mu_plot(ax, nom.sig_hi, nom.sig_lo, vals["c"], vals["d"], vals["u"], labels)
    fig.savefig("stability-tests-camps.pdf")

    nom, names, labels, vals = reg_stab_test(umbrella)
    fig, ax = plt.subplots(figsize=(5.2, 1.5 + len(names) * 0.315))
    fig.subplots_adjust(left=0.350, right=0.925, bottom=0.3, top=0.99)
    make_delta_mu_plot(ax, nom.sig_hi, nom.sig_lo, vals["c"], vals["d"], vals["u"], labels)
    fig.savefig("stability-tests-regions.pdf")

    fig, ax = b0_by_year(umbrella)
    fig.subplots_adjust(left=0.350, right=0.925, bottom=0.3, top=0.8)
    fig.savefig("stability-tests-b0-check.pdf")


if __name__ == "__main__":
    main()
