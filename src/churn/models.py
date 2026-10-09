"""Model zoo, hyper-parameter search spaces and ensembles.

Every model is a full ``Pipeline(FeatureEngineer -> [preprocessing] -> estimator)``
that consumes a *raw* customer DataFrame, so the saved artefact is the entire
inference path - no train/serve skew.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .deep import TabularNetClassifier
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, FeatureEngineer, make_preprocessor

SEED = 42
# Small data: few threads per model is much faster than n_jobs=-1 (OpenMP contention made
# LightGBM ~100x slower on a 14-thread laptop). Parallelism comes from concurrent Optuna trials.
THREADS = 1

DISPLAY_NAMES = {
    "dummy": "Baseline (majority)",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "tabular_net": "TabularNet (DL)",
    "stacked_ensemble": "Stacked Ensemble",
    "soft_voting": "Soft-Voting Ensemble",
}
TUNABLE = ["logistic_regression", "random_forest", "xgboost", "lightgbm", "catboost", "tabular_net"]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _estimator(name: str, params: dict[str, Any]):
    p = dict(params)
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    if name == "logistic_regression":
        l1 = p.pop("l1_ratio", 0.0)
        return LogisticRegression(
            l1_ratio=l1, solver="saga" if l1 > 0 else "lbfgs", max_iter=5000, random_state=SEED, **p
        )
    if name == "random_forest":
        return RandomForestClassifier(n_jobs=2, random_state=SEED, **p)
    if name == "xgboost":
        return XGBClassifier(
            tree_method="hist", eval_metric="logloss", n_jobs=THREADS, random_state=SEED, verbosity=0, **p
        )
    if name == "lightgbm":
        return LGBMClassifier(subsample_freq=1, n_jobs=THREADS, random_state=SEED, verbose=-1, **p)
    if name == "catboost":
        return CatBoostClassifier(
            # native categorical handling; one-hot inside CatBoost because every feature has <= 5
            # levels - CTR target statistics were ~18x slower here for no measurable AUC gain
            cat_features=CATEGORICAL_FEATURES, one_hot_max_size=8, random_seed=SEED, verbose=0,
            thread_count=2, allow_writing_files=False, **p,
        )
    if name == "tabular_net":
        if "hidden" in p and isinstance(p["hidden"], str):
            p["hidden"] = tuple(int(h) for h in p["hidden"].split("-"))
        return TabularNetClassifier(
            cat_cols=tuple(CATEGORICAL_FEATURES), num_cols=tuple(NUMERIC_FEATURES), random_state=SEED,
            n_threads=2, **p
        )
    raise ValueError(f"unknown model {name}")


def build_pipeline(name: str, params: dict[str, Any] | None = None) -> Pipeline:
    params = params or {}
    steps: list[tuple[str, Any]] = [("features", FeatureEngineer())]
    if name in ("dummy", "logistic_regression"):
        steps.append(("preprocess", make_preprocessor(scale=True)))
    elif name in ("random_forest", "xgboost", "lightgbm"):
        steps.append(("preprocess", make_preprocessor(scale=False)))
    # CatBoost handles categoricals natively; TabularNet learns embeddings.
    steps.append(("model", _estimator(name, params)))
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Optuna search spaces
# ---------------------------------------------------------------------------
def _lr_space(t: optuna.Trial) -> dict:
    return {
        "C": t.suggest_float("C", 1e-3, 10.0, log=True),
        "l1_ratio": t.suggest_categorical("l1_ratio", [0.0, 0.25, 0.5, 1.0]),
        "class_weight": t.suggest_categorical("class_weight", [None, "balanced"]),
    }


def _rf_space(t: optuna.Trial) -> dict:
    return {
        "n_estimators": t.suggest_int("n_estimators", 200, 800, step=100),
        "max_depth": t.suggest_int("max_depth", 4, 20),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 30),
        "max_features": t.suggest_float("max_features", 0.1, 0.8),
        "class_weight": t.suggest_categorical("class_weight", [None, "balanced_subsample"]),
    }


def _xgb_space(t: optuna.Trial) -> dict:
    return {
        "n_estimators": t.suggest_int("n_estimators", 100, 1200, step=50),
        "learning_rate": t.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "max_depth": t.suggest_int("max_depth", 2, 7),
        "min_child_weight": t.suggest_float("min_child_weight", 1, 30, log=True),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.3, 1.0),
        "gamma": t.suggest_float("gamma", 1e-3, 5.0, log=True),
        "reg_alpha": t.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
    }


def _lgbm_space(t: optuna.Trial) -> dict:
    return {
        "n_estimators": t.suggest_int("n_estimators", 100, 1200, step=50),
        "learning_rate": t.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "num_leaves": t.suggest_int("num_leaves", 4, 64, log=True),
        "max_depth": t.suggest_int("max_depth", 2, 10),
        "min_child_samples": t.suggest_int("min_child_samples", 10, 150, log=True),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": t.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
    }


def _cat_space(t: optuna.Trial) -> dict:
    return {
        "iterations": t.suggest_int("iterations", 200, 1000, step=100),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": t.suggest_int("depth", 3, 8),
        "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "random_strength": t.suggest_float("random_strength", 0.1, 5.0, log=True),
        "bagging_temperature": t.suggest_float("bagging_temperature", 0.0, 1.0),
    }


def _net_space(t: optuna.Trial) -> dict:
    return {
        "hidden": t.suggest_categorical("hidden", ["64", "128-64", "256-128", "128-64-32", "256-128-64"]),
        "dropout": t.suggest_float("dropout", 0.0, 0.5),
        "lr": t.suggest_float("lr", 3e-4, 1e-2, log=True),
        "weight_decay": t.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": t.suggest_categorical("batch_size", [128, 256, 512]),
        "emb_scale": t.suggest_float("emb_scale", 0.5, 2.0),
    }


SEARCH_SPACES: dict[str, Callable[[optuna.Trial], dict]] = {
    "logistic_regression": _lr_space,
    "random_forest": _rf_space,
    "xgboost": _xgb_space,
    "lightgbm": _lgbm_space,
    "catboost": _cat_space,
    "tabular_net": _net_space,
}

# Sensible starting points, enqueued as trial 0 so tuning never does worse than defaults
DEFAULT_PARAMS: dict[str, dict] = {
    "logistic_regression": {"C": 1.0, "l1_ratio": 0.0, "class_weight": None},
    "random_forest": {"n_estimators": 500, "max_depth": 10, "min_samples_leaf": 5,
                      "max_features": 0.3, "class_weight": None},
    "xgboost": {"n_estimators": 400, "learning_rate": 0.03, "max_depth": 3, "min_child_weight": 5,
                "subsample": 0.8, "colsample_bytree": 0.7, "gamma": 0.1, "reg_alpha": 0.01,
                "reg_lambda": 1.0},
    "lightgbm": {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 8, "max_depth": 4,
                 "min_child_samples": 40, "subsample": 0.8, "colsample_bytree": 0.7,
                 "reg_alpha": 0.01, "reg_lambda": 1.0},
    "catboost": {"iterations": 600, "learning_rate": 0.03, "depth": 5, "l2_leaf_reg": 5.0,
                 "random_strength": 1.0, "bagging_temperature": 0.5},
    "tabular_net": {"hidden": "128-64", "dropout": 0.2, "lr": 2e-3, "weight_decay": 1e-4,
                    "batch_size": 256, "emb_scale": 1.0},
}


# ---------------------------------------------------------------------------
# Ensembles (fitted base pipelines + a combiner)
# ---------------------------------------------------------------------------
def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class StackedEnsemble(ClassifierMixin, BaseEstimator):
    """Level-2 logistic regression over the base models' log-odds.

    Base models are fitted on the full training set; the meta-learner is fitted
    on their *out-of-fold* predictions so it never sees leaked scores. Stacking
    on log-odds with a logistic meta-learner also yields well-calibrated output.
    """

    def __init__(self, base_models: dict[str, Pipeline] | None = None, meta: LogisticRegression | None = None):
        self.base_models = base_models
        self.meta = meta

    def base_matrix(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([m.predict_proba(X)[:, 1] for m in self.base_models.values()])

    def fit_meta(self, oof_matrix: np.ndarray, y) -> "StackedEnsemble":
        self.meta = LogisticRegression(C=1.0, max_iter=1000).fit(_logit(oof_matrix), y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.meta.predict_proba(_logit(self.base_matrix(X)))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def weights(self) -> dict[str, float]:
        return dict(zip(self.base_models, self.meta.coef_[0].round(4)))


class SoftVotingEnsemble(ClassifierMixin, BaseEstimator):
    """Plain average of base-model probabilities."""

    def __init__(self, base_models: dict[str, Pipeline] | None = None):
        self.base_models = base_models
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.base_models.values()], axis=0)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
