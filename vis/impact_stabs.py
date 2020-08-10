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

ws704 = "/Users/ddavis/20200806/h704/rpcc_main_asimov/tW"
ws713 = "/Users/ddavis/20200806/h713/rpcc_main_asimov/tW"

ws704_data = "/Users/ddavis/20200806/h704/rpcc_main_data/tW"
ws713_data = "/Users/ddavis/20200806/h713/rpcc_main_data/tW"

print(tr.delta_mu(ws704, ws713))
print(tr.delta_mu(ws704_data, ws713_data))

print(tr.delta_mu(ws704_data, ws713_data, poi_name="mu_ttbar"))




nps = sorted([
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
], key=lambda n: n.post_max)

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
ax.text(0.10, 0.95, "ATLAS", fontstyle="italic", fontweight="bold", size=14, transform=ax.transAxes)
ax.text(0.37, 0.95, "Internal", size=14, transform=ax.transAxes)
ax.text(0.10, 0.92, "$\\sqrt{s}$ = 13 TeV, $L = {139}$ fb$^{-1}$", size=12, transform=ax.transAxes)
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
ax.text(0.10, 0.95, "ATLAS", fontstyle="italic", fontweight="bold", size=14, transform=ax.transAxes)
ax.text(0.37, 0.95, "Internal", size=14, transform=ax.transAxes)
ax.text(0.10, 0.92, "$\\sqrt{s}$ = 13 TeV, $L = {139}$ fb$^{-1}$", size=12, transform=ax.transAxes)
fig.subplots_adjust(left=0.45, bottom=0.085, top=0.915, right=0.975)
mpl_dir = PosixPath(".") / "matplotlib"
mpl_dir.mkdir(exist_ok=True)
fig.savefig(mpl_dir / "main_vs_main_unsorted.pdf")
plt.close(fig)
del fig, ax, ax2


def shotgun_1j1b(poi, h7v):
    os.chdir(f"/Users/ddavis/20200806/{h7v}")
    setups = [
        "asimov",
        "asimov_1j1b",
        "asimov_1j1b2j1b",
        "asimov_1j1b2j2b",
        "data",
        "data_1j1b",
        "data_1j1b2j1b",
        "data_1j1b2j2b",
        "data_only1516",
        "data_only17",
        "data_only18",
    ]
    nps = sorted([tr.nuispar_impact(f"rpcc_main_{s}/tW", poi, s) for s in setups], key=lambda n: n.post_max)
    os.chdir(starting_dir)
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
    ax.text(0.10, 0.95, "ATLAS", fontstyle="italic", fontweight="bold", size=14, transform=ax.transAxes)
    ax.text(0.37, 0.95, "Internal", size=14, transform=ax.transAxes)
    ax.text(0.10, 0.92, f"{poi}, {h7v}", size=14, transform=ax.transAxes)
    fig.subplots_adjust(left=0.45, bottom=0.085, top=0.915, right=0.975)
    mpl_dir = PosixPath(".") / "matplotlib"
    mpl_dir.mkdir(exist_ok=True)
    fig.savefig(mpl_dir / f"{poi}_{h7v}.pdf")
    plt.close(fig)
    del fig, ax, ax2


def shotgun_2j1b(poi, h7v):
    os.chdir(f"/Users/ddavis/20200806/{h7v}")
    setups = [
        "asimov",
        "asimov_2j1b",
        "asimov_1j1b2j1b",
        "data",
        "data_2j1b",
        "data_1j1b2j1b",
        "data_only1516",
        "data_only17",
        "data_only18",
    ]
    nps = sorted([tr.nuispar_impact(f"rpcc_main_{s}/tW", poi, s) for s in setups], key=lambda n: n.post_max)
    os.chdir(starting_dir)
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
    ax.text(0.10, 0.95, "ATLAS", fontstyle="italic", fontweight="bold", size=14, transform=ax.transAxes)
    ax.text(0.37, 0.95, "Internal", size=14, transform=ax.transAxes)
    ax.text(0.10, 0.92, f"{poi}, {h7v}", size=14, transform=ax.transAxes)
    fig.subplots_adjust(left=0.45, bottom=0.085, top=0.915, right=0.975)
    mpl_dir = PosixPath(".") / "matplotlib"
    mpl_dir.mkdir(exist_ok=True)
    fig.savefig(mpl_dir / f"{poi}_{h7v}.pdf")
    plt.close(fig)
    del fig, ax, ax2


def shotgun_2j2b(poi, h7v):
    os.chdir(f"/Users/ddavis/20200806/{h7v}")
    setups = [
        "asimov",
        "asimov_2j2b",
        "asimov_1j1b2j2b",
        "data",
        "data_2j2b",
        "data_1j1b2j2b",
        "data_only1516",
        "data_only17",
        "data_only18",
    ]
    nps = sorted([tr.nuispar_impact(f"rpcc_main_{s}/tW", poi, s) for s in setups], key=lambda n: n.post_max)
    os.chdir(starting_dir)
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
    ax.text(0.10, 0.95, "ATLAS", fontstyle="italic", fontweight="bold", size=14, transform=ax.transAxes)
    ax.text(0.37, 0.95, "Internal", size=14, transform=ax.transAxes)
    ax.text(0.10, 0.92, f"{poi}, {h7v}", size=14, transform=ax.transAxes)
    fig.subplots_adjust(left=0.45, bottom=0.085, top=0.915, right=0.975)
    mpl_dir = PosixPath(".") / "matplotlib"
    mpl_dir.mkdir(exist_ok=True)
    fig.savefig(mpl_dir / f"{poi}_{h7v}.pdf")
    plt.close(fig)
    del fig, ax, ax2


def shotgun_nm(poi, h7v):
    os.chdir(f"/Users/ddavis/20200806/{h7v}")
    setups = [
        "asimov",
        "asimov_1j1b",
        "asimov_2j1b",
        "asimov_2j2b",
        "asimov_1j1b2j1b",
        "asimov_1j1b2j2b",
        "data",
        "data_1j1b",
        "data_2j1b",
        "data_2j2b",
        "data_1j1b2j1b",
        "data_1j1b2j2b",
        "data_only1516",
        "data_only17",
        "data_only18",
    ]
    nps = sorted([tr.nuispar_impact(f"rpcc_main_{s}/tW", poi, s) for s in setups], key=lambda n: n.post_max)
    os.chdir(starting_dir)
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
    ax.text(0.10, 0.95, "ATLAS", fontstyle="italic", fontweight="bold", size=14, transform=ax.transAxes)
    ax.text(0.37, 0.95, "Internal", size=14, transform=ax.transAxes)
    ax.text(0.10, 0.92, f"{poi}, {h7v}", size=14, transform=ax.transAxes)
    fig.subplots_adjust(left=0.45, bottom=0.085, top=0.915, right=0.975)
    mpl_dir = PosixPath(".") / "matplotlib"
    mpl_dir.mkdir(exist_ok=True)
    fig.savefig(mpl_dir / f"{poi}_{h7v}.pdf")
    plt.close(fig)
    del fig, ax, ax2


shotgun_1j1b("ttbar_PS_1j1b", "h704")
shotgun_1j1b("ttbar_PS_1j1b", "h713")
shotgun_1j1b("tW_PS_1j1b", "h704")
shotgun_1j1b("tW_PS_1j1b", "h713")

shotgun_2j1b("ttbar_PS_2j1b", "h704")
shotgun_2j1b("ttbar_PS_2j1b", "h713")
shotgun_2j1b("tW_PS_2j1b", "h704")
shotgun_2j1b("tW_PS_2j1b", "h713")

shotgun_2j2b("ttbar_PS_2j2b", "h704")
shotgun_2j2b("ttbar_PS_2j2b", "h713")
shotgun_2j2b("tW_PS_2j2b", "h704")
shotgun_2j2b("tW_PS_2j2b", "h713")

shotgun_nm("ttbar_PS_norm", "h704")
shotgun_nm("ttbar_PS_norm", "h713")
shotgun_nm("tW_PS_norm", "h704")
shotgun_nm("tW_PS_norm", "h713")

shotgun_nm("ttbar_PS_migration", "h704")
shotgun_nm("ttbar_PS_migration", "h713")
shotgun_nm("tW_PS_migration", "h704")
shotgun_nm("tW_PS_migration", "h713")
