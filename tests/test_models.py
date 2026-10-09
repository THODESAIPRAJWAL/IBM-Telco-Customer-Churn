import numpy as np
import pickle

from churn.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, FeatureEngineer
from churn.deep import TabularNetClassifier
from churn.models import DEFAULT_PARAMS, build_pipeline


def test_tabular_net_fit_predict_and_unseen_category(df):
    X = FeatureEngineer().transform(df.iloc[:800])
    y = df["Churn"].iloc[:800].to_numpy()
    net = TabularNetClassifier(tuple(CATEGORICAL_FEATURES), tuple(NUMERIC_FEATURES), max_epochs=5, n_threads=1)
    net.fit(X, y)
    p = net.predict_proba(X.iloc[:10])
    assert p.shape == (10, 2) and np.allclose(p.sum(axis=1), 1)
    unseen = X.iloc[:2].copy()
    unseen["Contract"] = "Ten year"  # unseen level -> reserved index 0, must not crash
    assert net.predict_proba(unseen).shape == (2, 2)
    clone = pickle.loads(pickle.dumps(net))
    assert np.allclose(clone.predict_proba(X.iloc[:10]), p)


def test_every_pipeline_consumes_raw_frame(df):
    X = df.drop(columns=["Churn"]).iloc[:600]
    y = df["Churn"].iloc[:600]
    for name in ["logistic_regression", "lightgbm", "xgboost"]:
        pipe = build_pipeline(name, DEFAULT_PARAMS[name]).fit(X, y)
        p = pipe.predict_proba(X.iloc[:5])[:, 1]
        assert ((p >= 0) & (p <= 1)).all(), name
