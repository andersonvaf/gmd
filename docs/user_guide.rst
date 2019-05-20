.. title:: User guide

.. _user_guide:

==============
Usage Examples
==============

Use subspaces as LoF input
--------------------------

::

    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    import sklearn.metrics as metrics
    from sklearn.datasets import kddcup99
    from sklearn.neighbors import LocalOutlierFactor
    from gmd import GMD

    kdd = kddcup99.fetch_kddcup99(subset='SA')
    df = pd.DataFrame(kdd.data)
    df[[1,2,3]] = df[[1,2,3]].apply(LabelEncoder().fit_transform)
    df = df.apply(lambda x : pd.to_numeric(x))
    y_true = kdd.target != b'normal.'

    gmd = GMD()
    gmd.fit(df)

    subspaces = gmd.subspaces_
    preds = np.zeros((df.shape[0],len(subspaces)))
    for k, v in subspaces.items():
        clf.fit(df.iloc[:,subspaces[k]])
        preds[:,k] = clf.negative_outlier_factor_
    metrics.roc_auc_score(y_true, preds.sum(axis=1)*-1)
