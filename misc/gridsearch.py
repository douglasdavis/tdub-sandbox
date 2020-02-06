from tdub import setup_logging
from tdub.train import prepare_from_root
from tdub.utils import get_selection, get_features, quick_files

import lightgbm as lgbm
from sklearn.model_selection import train_test_split

from dask_jobqueue import HTCondorCluster
from dask.distributed import Client
from dask_ml.model_selection import GridSearchCV


cluster = HTCondorCluster(cores=2, disk="4GB", memory="8GB")
client = Client(cluster)
cluster.adapt(maximum_jobs=200)

setup_logging()

qf = quick_files("/atlasgpfs01/usatlas/data/ddavis/wtloop/v29_20191111")

df, y, w = prepare_from_root(qf["tW_DR"], qf["ttbar"], "1j1b")

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    df, y, w, train_size=0.8, random_state=414, shuffle=True
)

n_sig = y_train[y_train == 1].shape[0]
n_bkg = y_train[y_train == 0].shape[0]
spw = n_bkg / n_sig

n_sig = y[y == 1].shape[0]
n_bkg = y[y == 0].shape[0]
spw = n_bkg / n_sig
print(spw)

search_parameters = {
    "learning_rate": [0.02, 0.05, 0.1],
    "num_leaves": [20, 50, 150, 200],
    "min_child_samples": [40, 60, 100, 160, 240],
    "max_depth": [3, 4, 5, 6, 7, 8],
}

clf = lgbm.LGBMClassifier(boosting_type="gbdt", scale_pos_weight=spw, n_estimators=1000)

fit_params = {
    "early_stopping_rounds": 15,
    "eval_metric": "auc",
    "eval_set": [(X_test, y_test)],
    "eval_sample_weight": [w_test],
}

search = GridSearchCV(clf, param_grid=search_parameters, cv=2)
search.fit(df, y, **fit_params)
