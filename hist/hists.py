from tdub.frames import raw_dataframe
from tdub.utils import quick_files, files_for_tree, get_selection
from tdub.hist import generate_from_df, df2th1
import uproot

qf = quick_files("/Users/ddavis/ATLAS/data/wtloop/v29_20200201")

samples = [
    "Data",
    "ttbar",
    "tW_DR",
    "Zjets",
    "Diboson",
    "MCNP",
    #"tW_DR_AFII",
    #"tW_DS",
    #"ttbar_AFII",
    #"ttbar_PS",
    #"ttbar_hdamp",
]

all_histograms = {}

for samp in samples:
    print(f"working on {samp}")
    files = qf[samp]
    df = raw_dataframe(files, branches=["met", "weight_nominal", "reg1j1b", "reg2j1b", "reg2j2b", "elmu", "OS"])
    for region in ("reg1j1b", "reg2j1b", "reg2j2b"):
        sel = get_selection(region)
        dfc, dfe = generate_from_df(df.query(sel), "met", bins=15, range=(0.0, 200.0), systematic_weights=False)
        hists = df2th1(dfc, dfe, weight_col="weight_nominal")
        for hname, hobj in hists.items():
            if hname == "weight_nominal":
                finalkey = f"{region}_met_{samp}"
            else:
                finalkey = f"{region}_met_{samp}_{hname}"
            all_histograms[finalkey] = hobj


with uproot.recreate("hgrams.root") as f:
    for k, v in all_histograms.items():
        f[k] = v
