#!/usr/bin/env python

import pathlib
import json
import yaml
import sys

with open("meta.yml") as f:
    meta = yaml.load(f, Loader=yaml.Loader)
titles = meta["titles"]

prefix = r"""
\begin{tabular}{rlc}
  \toprule
  & Variable & Importance \\
  \midrule"""
single = """   {} & {} & {} \\\\"""
suffix = r""" \bottomrule
\end{tabular}}"""


class Pair(object):
    def __init__(self, f, i):
        self.f = f
        self.i = i


def pairs(fold):
    fold_dir = pathlib.PosixPath(fold)

    with open(fold_dir / "summary.json","r") as f:
        j = json.load(f)

    features = j["features"]
    importances = j["importances"]

    pairs = []
    for feat, imp in zip(features, importances):
        pairs.append(Pair(feat, imp))

    return sorted(pairs, key=lambda p: -p.i)


if __name__ == "__main__":
    print(prefix)
    for i, p in enumerate(pairs(sys.argv[1])):
        print(single.format(i + 1, titles[p.f]["mpl"], round(p.i, 3)))
    print(suffix)
