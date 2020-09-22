#!/usr/bin/env python

# stdlib
import os
from pathlib import PosixPath

# third party
import matplotlib.pyplot as plt
import numpy as np

# tdub
import tdub.rex as tr
from tdub.art import draw_impact_barh


starting_dir = PosixPath.cwd()

ws704 = "/Users/ddavis/Desktop/standardfits/704/rexpy-condor-main/tW"
ws713 = "/Users/ddavis/Desktop/standardfits/713/rexpy-condor-main/tW"


nps = sorted(
    [
        tr.nuispar_impact(ws704, "ttbar_PS_1j1b", "h704 ttbar PS 1j1b"),
        tr.nuispar_impact(ws713, "ttbar_PS_1j1b", "h713 ttbar PS 1j1b"),
        tr.nuispar_impact(ws704, "ttbar_PS_2j1b", "h704 ttbar PS 2j1b"),
        tr.nuispar_impact(ws713, "ttbar_PS_2j1b", "h713 ttbar PS 2j1b"),
        tr.nuispar_impact(ws704, "ttbar_PS_2j2b", "h704 ttbar PS 2j2b"),
        tr.nuispar_impact(ws713, "ttbar_PS_2j2b", "h713 ttbar PS 2j2b"),
        tr.nuispar_impact(ws704, "ttbar_PS_norm", "h704 ttbar PS norm"),
        tr.nuispar_impact(ws713, "ttbar_PS_norm", "h713 ttbar PS norm"),
        tr.nuispar_impact(ws704, "ttbar_PS_migration", "h704 ttbar PS migration"),
        tr.nuispar_impact(ws713, "ttbar_PS_migration", "h713 ttbar PS migration"),
        tr.nuispar_impact(ws704, "tW_PS_1j1b", "h704 tW PS 1j1b"),
        tr.nuispar_impact(ws713, "tW_PS_1j1b", "h713 tW PS 1j1b"),
        tr.nuispar_impact(ws704, "tW_PS_2j1b", "h704 tW PS 2j1b"),
        tr.nuispar_impact(ws713, "tW_PS_2j1b", "h713 tW PS 2j1b"),
        tr.nuispar_impact(ws704, "tW_PS_2j2b", "h704 tW PS 2j2b"),
        tr.nuispar_impact(ws713, "tW_PS_2j2b", "h713 tW PS 2j2b"),
        tr.nuispar_impact(ws704, "tW_PS_norm", "h704 tW PS norm"),
        tr.nuispar_impact(ws713, "tW_PS_norm", "h713 tW PS norm"),
        tr.nuispar_impact(ws704, "tW_PS_migration", "h704 tW PS migration"),
        tr.nuispar_impact(ws713, "tW_PS_migration", "h713 tW PS migration"),
    ],
    key=lambda n: n.post_max,
)

df = tr.nuispar_impact_plot_df(nps)
ys = np.array(df.ys)
fig, ax = plt.subplots(figsize=(5, 8.5))
ax, ax2 = draw_impact_barh(ax, df)
ax.legend(ncol=1, loc="upper left", bbox_to_anchor=(-0.75, 1.11))
ax.set_xticks([-0.2, -0.1, 0.0, 0.1, 0.2])
ax.set_ylim([-1, ys[-1] + 2.4])
ax.set_yticklabels([p.label for p in nps])
ax2.legend(loc="lower left", bbox_to_anchor=(-0.75, -0.09))
ax2.set_xlabel(r"$\Delta\mu$", labelpad=25)
ax.set_xlabel(r"$(\hat{\theta}-\theta_0)/\Delta\theta$", labelpad=20)
ax.text(
    0.10,
    0.95,
    "ATLAS",
    fontstyle="italic",
    fontweight="bold",
    size=14,
    transform=ax.transAxes,
)
ax.text(0.37, 0.95, "Internal", size=14, transform=ax.transAxes)
ax.text(
    0.10,
    0.92,
    "$\\sqrt{s}$ = 13 TeV, $L = {139}$ fb$^{-1}$",
    size=12,
    transform=ax.transAxes,
)
fig.subplots_adjust(left=0.45, bottom=0.085, top=0.915, right=0.975)
mpl_dir = PosixPath(".") / "matplotlib"
mpl_dir.mkdir(exist_ok=True)
fig.savefig(mpl_dir / "main_vs_main.pdf")
plt.close(fig)
del fig, ax, ax2


nps = [
    tr.nuispar_impact(ws704, "ttbar_PS_1j1b", "h704 ttbar PS 1j1b"),
    tr.nuispar_impact(ws713, "ttbar_PS_1j1b", "h713 ttbar PS 1j1b"),
    tr.nuispar_impact(ws704, "ttbar_PS_2j1b", "h704 ttbar PS 2j1b"),
    tr.nuispar_impact(ws713, "ttbar_PS_2j1b", "h713 ttbar PS 2j1b"),
    tr.nuispar_impact(ws704, "ttbar_PS_2j2b", "h704 ttbar PS 2j2b"),
    tr.nuispar_impact(ws713, "ttbar_PS_2j2b", "h713 ttbar PS 2j2b"),
    tr.nuispar_impact(ws704, "ttbar_PS_norm", "h704 ttbar PS norm"),
    tr.nuispar_impact(ws713, "ttbar_PS_norm", "h713 ttbar PS norm"),
    tr.nuispar_impact(ws704, "ttbar_PS_migration", "h704 ttbar PS migration"),
    tr.nuispar_impact(ws713, "ttbar_PS_migration", "h713 ttbar PS migration"),
    tr.nuispar_impact(ws704, "tW_PS_1j1b", "h704 tW PS 1j1b"),
    tr.nuispar_impact(ws713, "tW_PS_1j1b", "h713 tW PS 1j1b"),
    tr.nuispar_impact(ws704, "tW_PS_2j1b", "h704 tW PS 2j1b"),
    tr.nuispar_impact(ws713, "tW_PS_2j1b", "h713 tW PS 2j1b"),
    tr.nuispar_impact(ws704, "tW_PS_2j2b", "h704 tW PS 2j2b"),
    tr.nuispar_impact(ws713, "tW_PS_2j2b", "h713 tW PS 2j2b"),
    tr.nuispar_impact(ws704, "tW_PS_norm", "h704 tW PS norm"),
    tr.nuispar_impact(ws713, "tW_PS_norm", "h713 tW PS norm"),
    tr.nuispar_impact(ws704, "tW_PS_migration", "h704 tW PS migration"),
    tr.nuispar_impact(ws713, "tW_PS_migration", "h713 tW PS migration"),
]

df = tr.nuispar_impact_plot_df(nps)
ys = np.array(df.ys)
fig, ax = plt.subplots(figsize=(5, 8.5))
ax, ax2 = draw_impact_barh(ax, df)
ax.legend(ncol=1, loc="upper left", bbox_to_anchor=(-0.75, 1.11))
ax.set_xticks([-0.2, -0.1, 0.0, 0.1, 0.2])
ax.set_ylim([-1, ys[-1] + 2.4])
ax.set_yticklabels([p.label for p in nps])
ax2.legend(loc="lower left", bbox_to_anchor=(-0.75, -0.09))
ax2.set_xlabel(r"$\Delta\mu$", labelpad=25)
ax.set_xlabel(r"$(\hat{\theta}-\theta_0)/\Delta\theta$", labelpad=20)
ax.text(
    0.10,
    0.95,
    "ATLAS",
    fontstyle="italic",
    fontweight="bold",
    size=14,
    transform=ax.transAxes,
)
ax.text(0.37, 0.95, "Internal", size=14, transform=ax.transAxes)
ax.text(
    0.10,
    0.92,
    "$\\sqrt{s}$ = 13 TeV, $L = {139}$ fb$^{-1}$",
    size=12,
    transform=ax.transAxes,
)
fig.subplots_adjust(left=0.45, bottom=0.085, top=0.915, right=0.975)
mpl_dir = PosixPath(".") / "matplotlib"
mpl_dir.mkdir(exist_ok=True)
fig.savefig(mpl_dir / "main_vs_main_unsorted.pdf")
plt.close(fig)
del fig, ax, ax2


def r1j1b(poi, h7v):
    os.chdir(f"/Users/ddavis/Desktop/standardfits/{h7v}")
    configs = reversed([
        ("main", "Complete"),
        ("main_1j1b", "1j1b Only"),
        ("main_1j1b2j1b", "1j1b + 2j1b"),
        ("main_1j1b2j2b", "1j1b + 2j2b"),
        ("main_only1516", "201(5,6)/MC16a only"),
        ("main_only17", "2017/MC16d only"),
        ("main_only18", "2018/MC16e only"),
    ])
    nps = [tr.nuispar_impact(f"rexpy-condor-{sd}/tW", poi, sl) for sd, sl in configs]
    os.chdir(starting_dir)
    df = tr.nuispar_impact_plot_df(nps)
    fig, ax = plt.subplots(figsize=(4, 5.5))
    ax, ax2 = draw_impact_barh(ax, df)
    ys = np.array(df.ys)
    ax.legend(ncol=1, loc="upper left", bbox_to_anchor=(-0.85, 1.11), fontsize="x-small")
    ax.set_xticks([-0.2, -0.1, 0.0, 0.1, 0.2])
    ax.set_ylim([-1, ys[-1] + 2.4])
    ax.set_yticklabels([p.label for p in nps], size="x-small")
    ax2.legend(loc="lower left", bbox_to_anchor=(-0.85, -0.1), fontsize="x-small")
    ax2.set_xlabel(r"$\Delta\mu$", labelpad=20, size="x-small")
    ax.set_xlabel(r"$(\hat{\theta}-\theta_0)/\Delta\theta$", labelpad=18, size="x-small")
    ax.text(
        0.05,
        0.95,
        "ATLAS",
        fontstyle="italic",
        fontweight="bold",
        size=12,
        transform=ax.transAxes,
    )
    ax.text(0.35, 0.95, "Internal", size="x-small", transform=ax.transAxes)
    ax.text(0.05, 0.915, "$\\sqrt{s}$ = 13 TeV", size="x-small", transform=ax.transAxes)
    ax.text(0.05, 0.88, f"NP: {poi}, H: {h7v}", size="x-small", transform=ax.transAxes)
    fig.subplots_adjust(left=0.45, bottom=0.085, top=0.915, right=0.975)
    mpl_dir = PosixPath(".") / "matplotlib"
    mpl_dir.mkdir(exist_ok=True)
    fig.savefig(mpl_dir / f"indivpoi_{poi}_{h7v}.pdf")
    plt.close(fig)
    del fig, ax, ax2
    return 0


def r2j1b(poi, h7v):
    os.chdir(f"/Users/ddavis/Desktop/standardfits/{h7v}")
    configs = reversed([
        ("main", "Complete"),
        ("main_2j1b", "2j1b Only"),
        ("main_1j1b2j1b", "1j1b + 2j1b"),
        ("main_only1516", "201(5,6)/MC16a only"),
        ("main_only17", "2017/MC16d only"),
        ("main_only18", "2018/MC16e only"),
    ])
    nps = [tr.nuispar_impact(f"rexpy-condor-{sd}/tW", poi, sl) for sd, sl in configs]
    os.chdir(starting_dir)
    df = tr.nuispar_impact_plot_df(nps)
    fig, ax = plt.subplots(figsize=(4, 5.25))
    ax, ax2 = draw_impact_barh(ax, df)
    ys = np.array(df.ys)
    ax.legend(ncol=1, loc="upper left", bbox_to_anchor=(-0.85, 1.11), fontsize="x-small")
    ax.set_xticks([-0.2, -0.1, 0.0, 0.1, 0.2])
    ax.set_ylim([-1, ys[-1] + 2.4])
    ax.set_yticklabels([p.label for p in nps], size="x-small")
    ax2.legend(loc="lower left", bbox_to_anchor=(-0.85, -0.1), fontsize="x-small")
    ax2.set_xlabel(r"$\Delta\mu$", labelpad=20, size="x-small")
    ax.set_xlabel(r"$(\hat{\theta}-\theta_0)/\Delta\theta$", labelpad=18, size="x-small")
    ax.text(
        0.05,
        0.95,
        "ATLAS",
        fontstyle="italic",
        fontweight="bold",
        size=12,
        transform=ax.transAxes,
    )
    ax.text(0.35, 0.95, "Internal", size="x-small", transform=ax.transAxes)
    ax.text(0.05, 0.915, "$\\sqrt{s}$ = 13 TeV", size="x-small", transform=ax.transAxes)
    ax.text(0.05, 0.88, f"NP: {poi}, H: {h7v}", size="x-small", transform=ax.transAxes)
    fig.subplots_adjust(left=0.45, bottom=0.085, top=0.915, right=0.975)
    mpl_dir = PosixPath(".") / "matplotlib"
    mpl_dir.mkdir(exist_ok=True)
    fig.savefig(mpl_dir / f"indivpoi_{poi}_{h7v}.pdf")
    plt.close(fig)
    del fig, ax, ax2
    return 0


def r2j2b(poi, h7v):
    os.chdir(f"/Users/ddavis/Desktop/standardfits/{h7v}")
    configs = reversed([
        ("main", "Complete"),
        ("main_2j2b", "2j2b Only"),
        ("main_1j1b2j2b", "1j1b + 2j2b"),
        ("main_only1516", "201(5,6)/MC16a only"),
        ("main_only17", "2017/MC16d only"),
        ("main_only18", "2018/MC16e only"),
    ])
    nps = [tr.nuispar_impact(f"rexpy-condor-{sd}/tW", poi, sl) for sd, sl in configs]
    os.chdir(starting_dir)
    df = tr.nuispar_impact_plot_df(nps)
    fig, ax = plt.subplots(figsize=(4, 5.25))
    ax, ax2 = draw_impact_barh(ax, df)
    ys = np.array(df.ys)
    ax.legend(ncol=1, loc="upper left", bbox_to_anchor=(-0.85, 1.11), fontsize="x-small")
    ax.set_xticks([-0.2, -0.1, 0.0, 0.1, 0.2])
    ax.set_ylim([-1, ys[-1] + 2.4])
    ax.set_yticklabels([p.label for p in nps], size="x-small")
    ax2.legend(loc="lower left", bbox_to_anchor=(-0.85, -0.1), fontsize="x-small")
    ax2.set_xlabel(r"$\Delta\mu$", labelpad=20, size="x-small")
    ax.set_xlabel(r"$(\hat{\theta}-\theta_0)/\Delta\theta$", labelpad=18, size="x-small")
    ax.text(
        0.05,
        0.95,
        "ATLAS",
        fontstyle="italic",
        fontweight="bold",
        size=12,
        transform=ax.transAxes,
    )
    ax.text(0.35, 0.95, "Internal", size="x-small", transform=ax.transAxes)
    ax.text(0.05, 0.915, "$\\sqrt{s}$ = 13 TeV", size="x-small", transform=ax.transAxes)
    ax.text(0.05, 0.88, f"NP: {poi}, H: {h7v}", size="x-small", transform=ax.transAxes)
    fig.subplots_adjust(left=0.45, bottom=0.085, top=0.915, right=0.975)
    mpl_dir = PosixPath(".") / "matplotlib"
    mpl_dir.mkdir(exist_ok=True)
    fig.savefig(mpl_dir / f"indivpoi_{poi}_{h7v}.pdf")
    plt.close(fig)
    del fig, ax, ax2
    return 0


def norm_mig(poi, h7v):
    os.chdir(f"/Users/ddavis/Desktop/standardfits/{h7v}")
    configs = reversed([
        ("main", "Complete"),
        ("main_1j1b", "1j1b Only"),
        ("main_2j1b", "2j1b Only"),
        ("main_2j2b", "2j2b Only"),
        ("main_1j1b2j1b", "1j1b + 2j1b"),
        ("main_1j1b2j2b", "1j1b + 2j2b"),
        ("main_only1516", "201(5,6)/MC16a only"),
        ("main_only17", "2017/MC16d only"),
        ("main_only18", "2018/MC16e only"),
    ])
    nps = [tr.nuispar_impact(f"rexpy-condor-{sd}/tW", poi, sl) for sd, sl in configs]
    os.chdir(starting_dir)
    df = tr.nuispar_impact_plot_df(nps)
    fig, ax = plt.subplots(figsize=(4, 5.5))
    ax, ax2 = draw_impact_barh(ax, df)
    ys = np.array(df.ys)
    ax.legend(ncol=1, loc="upper left", bbox_to_anchor=(-0.85, 1.11), fontsize="x-small")
    ax.set_xticks([-0.2, -0.1, 0.0, 0.1, 0.2])
    ax.set_ylim([-1, ys[-1] + 2.4])
    ax.set_yticklabels([p.label for p in nps], size="x-small")
    ax2.legend(loc="lower left", bbox_to_anchor=(-0.85, -0.1), fontsize="x-small")
    ax2.set_xlabel(r"$\Delta\mu$", labelpad=20, size="x-small")
    ax.set_xlabel(r"$(\hat{\theta}-\theta_0)/\Delta\theta$", labelpad=18, size="x-small")
    ax.text(
        0.05,
        0.95,
        "ATLAS",
        fontstyle="italic",
        fontweight="bold",
        size="x-small",
        transform=ax.transAxes,
    )
    ax.text(0.35, 0.95, "Internal", size="x-small", transform=ax.transAxes)
    ax.text(0.05, 0.915, "$\\sqrt{s}$ = 13 TeV", size="x-small", transform=ax.transAxes)
    ax.text(0.05, 0.88, f"NP: {poi}, H: {h7v}", size="x-small", transform=ax.transAxes)
    fig.subplots_adjust(left=0.45, bottom=0.085, top=0.915, right=0.975)
    mpl_dir = PosixPath(".") / "matplotlib"
    mpl_dir.mkdir(exist_ok=True)
    fig.savefig(mpl_dir / f"indivpoi_{poi}_{h7v}.pdf")
    plt.close(fig)
    del fig, ax, ax2
    return 0


r1j1b("ttbar_PS_1j1b", "704")
r1j1b("ttbar_PS_1j1b", "713")

r2j1b("ttbar_PS_2j1b", "704")
r2j1b("ttbar_PS_2j1b", "713")

r2j2b("ttbar_PS_2j2b", "704")
r2j2b("ttbar_PS_2j2b", "713")

r1j1b("tW_PS_1j1b", "704")
r1j1b("tW_PS_1j1b", "713")

r2j1b("tW_PS_2j1b", "704")
r2j1b("tW_PS_2j1b", "713")

r2j2b("tW_PS_2j2b", "704")
r2j2b("tW_PS_2j2b", "713")

norm_mig("tW_PS_norm", "704")
norm_mig("tW_PS_norm", "713")
norm_mig("tW_PS_migration", "704")
norm_mig("tW_PS_migration", "713")

norm_mig("ttbar_PS_norm", "704")
norm_mig("ttbar_PS_norm", "713")
norm_mig("ttbar_PS_migration", "704")
norm_mig("ttbar_PS_migration", "713")
