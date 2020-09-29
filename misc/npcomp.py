#!/usr/bin/env python

import tdub.rex as tr

fit1 = "/ddd/atlas/analysis/run/fitting/standards/h704/main.d/tW"
fit2 = "/ddd/atlas/analysis/run/fitting/standards/h704/orig.d/tW"

nps = ["ttbar_PS_2j1b"]

tr.comparison_summary(
    fit1,
    fit2,
    label1="with_additional_cut",
    label2="original_fit",
    fit_poi="SigXsecOverSM",
    nuispars=nps
)
