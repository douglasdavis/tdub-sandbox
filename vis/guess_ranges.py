#!/usr/bin/env python

import pathlib

import yaml
import click
import numpy as np

from tdub.utils import quick_files, get_selection
from tdub.frames import raw_dataframe

with open("meta.yml", "r") as f:
    META = yaml.load(f, Loader=yaml.Loader)

@click.command()
@click.argument("datadir")
def check(datadir: str):
    pairs = []
    allbranches = set()
    for reg, entries in META["regions"].items():
        for entry in entries:
            pairs.append((reg, entry["var"]))
            allbranches.add(entry["var"])
    allbranches.add("reg1j1b")
    allbranches.add("reg2j1b")
    allbranches.add("reg2j2b")
    allbranches.add("OS")
    allbranches.add("elmu")

    qf = quick_files(datadir)
    df = raw_dataframe(qf["Data"], branches=sorted(allbranches, key=str.lower))
    df1j1b = df.query(get_selection("1j1b"))
    df2j1b = df.query(get_selection("2j1b"))
    df2j2b = df.query(get_selection("2j2b"))

    for r, v in pairs:
        if r == "r1j1b":
            x = df1j1b[v]
            w = df1j1b["weight_nominal"]
        if r == "r2j1b":
            x = df2j1b[v]
            w = df2j1b["weight_nominal"]
        if r == "r2j2b":
            x = df2j2b[v]
            w = df2j2b["weight_nominal"]
        n, bins = np.histogram(x, bins=35)
        print(r, v, bins[0], bins[-1])


if __name__ == "__main__":
    check()
