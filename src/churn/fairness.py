"""Responsible-AI checks: group performance parity and the `gender` ablation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from . import plotting as P
from .config import TABLES_DIR
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, FeatureEngineer
from .models import _estimator

plt = P.plt


class _WithGender(FeatureEngineer):
    def transform(self, X):
        out = super().transform(X)
        out["gender_male"] = (X["gender"] == "Male").astype(int).to_numpy()
        return out


def group_metrics(X: pd.DataFrame, y, p, targeted, groups=("gender", "SeniorCitizen", "Partner", "Dependents")) -> pd.DataFrame:
    y, p, t = np.asarray(y), np.asarray(p), np.asarray(targeted, bool)
    rows = []
    for g in groups:
        for level in sorted(X[g].unique()):
            m = (X[g] == level).to_numpy()
            yy, tt = y[m], t[m]
            rows.append({
                "attribute": g,
                "group": ("Yes" if level == 1 else "No") if g == "SeniorCitizen" else str(level),
                "n": int(m.sum()),
                "base_churn_rate": yy.mean(),
                "selection_rate": tt.mean(),
                "tpr": tt[yy == 1].mean() if (yy == 1).any() else np.nan,
                "fpr": tt[yy == 0].mean() if (yy == 0).any() else np.nan,
                "precision": yy[tt].mean() if tt.any() else np.nan,
                "roc_auc": roc_auc_score(yy, p[m]) if len(set(yy)) > 1 else np.nan,
            })
    return pd.DataFrame(rows)


def disparity(gm: pd.DataFrame) -> pd.DataFrame:
    out = []
    for attr, g in gm.groupby("attribute"):
        out.append({
            "attribute": attr,
            "tpr_gap": g["tpr"].max() - g["tpr"].min(),
            "fpr_gap": g["fpr"].max() - g["fpr"].min(),
            "auc_gap": g["roc_auc"].max() - g["roc_auc"].min(),
            "selection_rate_ratio": g["selection_rate"].min() / g["selection_rate"].max(),
            "base_rate_ratio": g["base_churn_rate"].min() / g["base_churn_rate"].max(),
        })
    return pd.DataFrame(out)


def gender_ablation(params: dict, X, y, splits) -> dict:
    """CV AUC of LightGBM with vs without gender as an input."""
    def pipe(with_gender: bool):
        fe = _WithGender() if with_gender else FeatureEngineer()
        num = NUMERIC_FEATURES + (["gender_male"] if with_gender else [])
        pre = ColumnTransformer([("num", "passthrough", num),
                                 ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)])
        return Pipeline([("features", fe), ("preprocess", pre), ("model", _estimator("lightgbm", params))])

    res = {}
    for label, wg in [("without_gender", False), ("with_gender", True)]:
        aucs = []
        for tr, va in splits:
            m = pipe(wg).fit(X.iloc[tr], y.iloc[tr])
            aucs.append(roc_auc_score(y.iloc[va], m.predict_proba(X.iloc[va])[:, 1]))
        res[label] = {"cv_auc_mean": float(np.mean(aucs)), "cv_auc_std": float(np.std(aucs))}
    return res


def fig_group_parity(gm: pd.DataFrame) -> None:
    metrics = [("roc_auc", "ROC-AUC"), ("tpr", "True positive rate"), ("fpr", "False positive rate"),
               ("selection_rate", "Share targeted")]
    gm = gm.copy()
    gm["label"] = gm["attribute"] + ": " + gm["group"]
    order = gm["label"].tolist()[::-1]
    fig, axes = plt.subplots(1, 4, figsize=(16, 0.42 * len(gm) + 1.8), sharey=True, layout="constrained")
    for ax, (col, name) in zip(axes, metrics):
        d = gm.set_index("label").loc[order]
        for i, (lab, r) in enumerate(d.iterrows()):
            ax.plot(r[col], i, "o", color=P.BLUE, ms=7, markeredgecolor=P.SURFACE, markeredgewidth=1.5)
        # connect the groups of the same attribute to show the gap
        for attr, g in d.reset_index().groupby("attribute"):
            idx = [order.index(l) for l in g["label"]]
            ax.plot(g[col], idx, color=P.GHOST, lw=2, zorder=0)
        ax.set_title(name, fontsize=10.5)
        ax.set_yticks(range(len(order)), order, fontsize=9)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Fairness audit: model performance and targeting rates by group (test set)", x=0.01, ha="left",
                 fontsize=13, fontweight="semibold")
    P.savefig(fig, "fairness_group_parity")


def run(X_test, y_test, p_test, targeted, lgbm_params, X_train, y_train, splits) -> dict:
    gm = group_metrics(X_test, y_test, p_test, targeted)
    disp = disparity(gm)
    abl = gender_ablation(lgbm_params, X_train, y_train, splits)
    fig_group_parity(gm)
    gm.to_csv(TABLES_DIR / "fairness_group_metrics.csv", index=False)
    disp.to_csv(TABLES_DIR / "fairness_disparity.csv", index=False)
    return {"disparity": disp.round(4).to_dict("records"), "gender_ablation": abl}
