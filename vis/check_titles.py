#!/usr/bin/env python3

import yaml
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

with open("meta.yml", "r") as f:
    yam = yaml.load(f, Loader=yaml.Loader)

entries = [v["mpl"] for k, v in yam["titles"].items()]
units = [v["unit"] for k, v in yam["titles"].items()]

n = len(entries)

fig, ax = plt.subplots(figsize=(4, n/2.5))
for i, (entry, unit) in enumerate(zip(entries, units)):
    txt = f"{entry} [{unit}]"
    txt = txt.replace(" []", "")
    ax.text(0.2, i + .5, txt)
ax.set_ylim([0, n])
fig.savefig("pdfs/titles.pdf")
