"""Model training, tuning, selection and hold-out evaluation.

Protocol (designed to avoid optimistic bias):

1. 80/20 stratified split. The 20% test set is scored exactly once, at the end.
2. Every model family is tuned with Optuna (TPE sampler + median pruning) on the
   *same* 5-fold stratified CV splits of the training set.
3. Out-of-fold (OOF) predictions from the tuned models feed the ensembles, the
   model-selection decision (best CV AUC; ties within one standard error broken by
   OOF calibration) and the business-threshold optimisation.
4. The selected model is refit on the full training set and evaluated on test.
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import business
from . import evaluate as E
from .config import MODELS_DIR, TABLES_DIR, business_params, load_config
from .features import FeatureEngineer, make_preprocessor
from .models import (
    DEFAULT_PARAMS,
    DISPLAY_NAMES,
    SEARCH_SPACES,
    TUNABLE,
    SoftVotingEnsemble,
    StackedEnsemble,
    _estimator,
    _logit,
    build_pipeline,
)

log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)


def make_splits(X: pd.DataFrame, y: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    cfg = load_config()["split"]
    skf = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=cfg["random_state"])
    return list(skf.split(X, y))


def cross_validate(name: str, params: dict, X: pd.DataFrame, y: pd.Series, splits, trial=None):
    """OOF predictions + per-fold AUCs; reports to Optuna for pruning."""
    oof = np.zeros(len(y))
    fold_aucs = []
    for k, (tr, va) in enumerate(splits):
        pipe = build_pipeline(name, params)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = pipe.predict_proba(X.iloc[va])[:, 1]
        fold_aucs.append(roc_auc_score(y.iloc[va], oof[va]))
        if trial is not None:
            trial.report(float(np.mean(fold_aucs)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return oof, fold_aucs


def tune(name: str, X, y, splits, n_trials: int, timeout: int) -> tuple[dict, optuna.Study]:
    def objective(trial: optuna.Trial) -> float:
        params = SEARCH_SPACES[name](trial)
        _, aucs = cross_validate(name, params, X, y, splits, trial)
        return float(np.mean(aucs))

    study = optuna.create_study(
        direction="maximize",
        study_name=name,
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.enqueue_trial(DEFAULT_PARAMS[name])
    # Trials run concurrently in threads; the GBM / torch kernels release the GIL.
    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=load_config()["tuning"]["parallel_trials"],
                   gc_after_trial=True)
    return study.best_params, study


def imbalance_experiment(params: dict, X, y, splits) -> pd.DataFrame:
    """Does re-balancing help? Compare strategies on the tuned LightGBM."""
    neg_pos = float((y == 0).sum() / (y == 1).sum())
    strategies = {
        "None (natural 26.5% churn)": ("plain", {}),
        "Class weighting (scale_pos_weight)": ("plain", {"scale_pos_weight": neg_pos}),
        "SMOTE oversampling": ("smote", {}),
        "Random undersampling": ("under", {}),
    }
    rows = []
    for label, (kind, extra) in strategies.items():
        oof = np.zeros(len(y))
        for tr, va in splits:
            est = _estimator("lightgbm", {**params, **extra})
            steps = [("features", FeatureEngineer()), ("preprocess", make_preprocessor(scale=False))]
            if kind == "smote":
                steps.append(("resample", SMOTE(random_state=42)))
            elif kind == "under":
                steps.append(("resample", RandomUnderSampler(random_state=42)))
            pipe = ImbPipeline(steps + [("model", est)])
            pipe.fit(X.iloc[tr], y.iloc[tr])
            oof[va] = pipe.predict_proba(X.iloc[va])[:, 1]
        pred = (oof >= 0.5).astype(int)
        rows.append({
            "strategy": label,
            "roc_auc": roc_auc_score(y, oof),
            "pr_auc": average_precision_score(y, oof),
            "brier": brier_score_loss(y, oof),
            "mean_predicted_prob": float(oof.mean()),
            "recall_at_0.5": recall_score(y, pred),
            "precision_at_0.5": precision_score(y, pred),
        })
    return pd.DataFrame(rows)


def run(X_train, X_test, y_train, y_test, fast: bool = False, reuse_params: bool = False) -> dict[str, Any]:
    cfg = load_config()
    bp = business_params()
    splits = make_splits(X_train, y_train)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    params_file = MODELS_DIR / "best_params.json"
    saved = json.loads(params_file.read_text()) if (reuse_params and params_file.exists()) else {}

    oof: dict[str, np.ndarray] = {}
    test_pred: dict[str, np.ndarray] = {}
    fitted: dict[str, Any] = {}
    cv_rows, best_params, tuning_log = [], {}, []

    # --- Baseline ---------------------------------------------------------
    oof["dummy"], aucs = cross_validate("dummy", {}, X_train, y_train, splits)
    cv_rows.append({"key": "dummy", "cv_auc_mean": np.mean(aucs), "cv_auc_std": np.std(aucs), "fit_seconds": 0.0})

    # --- Tuned model families -------------------------------------------
    for name in TUNABLE:
        t0 = time.perf_counter()
        if name in saved:
            params = saved[name]
            n_done = 0
        else:
            n_trials = cfg["tuning"]["fast_n_trials"] if fast else cfg["tuning"]["n_trials"][name]
            params, study = tune(name, X_train, y_train, splits, n_trials, cfg["tuning"]["timeout_per_model_sec"])
            n_done = len(study.trials)
            tuning_log.append({"model": DISPLAY_NAMES[name], "trials": n_done,
                               "pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
                               "best_cv_auc": study.best_value})
        best_params[name] = params
        oof[name], aucs = cross_validate(name, params, X_train, y_train, splits)
        t1 = time.perf_counter()
        pipe = build_pipeline(name, params).fit(X_train, y_train)
        fit_s = time.perf_counter() - t1
        fitted[name] = pipe
        test_pred[name] = pipe.predict_proba(X_test)[:, 1]
        cv_rows.append({"key": name, "cv_auc_mean": np.mean(aucs), "cv_auc_std": np.std(aucs), "fit_seconds": fit_s})
        log.info("%-20s CV AUC %.4f +/- %.4f  (%d trials, %.0fs)", DISPLAY_NAMES[name], np.mean(aucs),
                 np.std(aucs), n_done, time.perf_counter() - t0)
    params_file.write_text(json.dumps(best_params, indent=2))

    # --- Ensembles --------------------------------------------------------
    base_keys = list(TUNABLE)
    oof_matrix = np.column_stack([oof[k] for k in base_keys])
    stack = StackedEnsemble({k: fitted[k] for k in base_keys}).fit_meta(oof_matrix, y_train)
    oof["stacked_ensemble"] = cross_val_predict(
        LogisticRegression(C=1.0, max_iter=1000), _logit(oof_matrix), y_train,
        cv=StratifiedKFold(5, shuffle=True, random_state=7), method="predict_proba")[:, 1]
    fitted["stacked_ensemble"] = stack
    test_pred["stacked_ensemble"] = stack.predict_proba(X_test)[:, 1]

    top3 = sorted(base_keys, key=lambda k: -roc_auc_score(y_train, oof[k]))[:3]
    vote = SoftVotingEnsemble({k: fitted[k] for k in top3})
    oof["soft_voting"] = np.mean([oof[k] for k in top3], axis=0)
    fitted["soft_voting"] = vote
    test_pred["soft_voting"] = vote.predict_proba(X_test)[:, 1]
    for key in ("stacked_ensemble", "soft_voting"):
        fold = [roc_auc_score(y_train.iloc[va], oof[key][va]) for _, va in splits]
        cv_rows.append({"key": key, "cv_auc_mean": np.mean(fold), "cv_auc_std": np.std(fold),
                        "fit_seconds": sum(r["fit_seconds"] for r in cv_rows if r["key"] in
                                           (base_keys if key == "stacked_ensemble" else top3))})

    cv = pd.DataFrame(cv_rows)
    cv["model"] = cv["key"].map(DISPLAY_NAMES)
    cv["oof_pr_auc"] = [average_precision_score(y_train, oof[k]) for k in cv["key"]]
    cv["oof_brier"] = [brier_score_loss(y_train, oof[k]) for k in cv["key"]]

    # --- Selection (on CV only - never on the test set) --------------------
    candidates = cv[cv["key"] != "dummy"]
    # 1) one-standard-error rule: models within 1 SE of the best CV AUC are statistically tied on ranking;
    # 2) among tied models deploy the best-calibrated one (lowest OOF Brier), because the campaign rule
    #    multiplies probabilities by dollar values.
    ranked = candidates.sort_values("cv_auc_mean", ascending=False)
    se = ranked.iloc[0]["cv_auc_std"] / np.sqrt(len(splits))
    tied = ranked[ranked["cv_auc_mean"] >= ranked.iloc[0]["cv_auc_mean"] - se]
    within_1se = tied["model"].tolist()
    selected = tied.sort_values("oof_brier").iloc[0]["key"]
    sel_name = DISPLAY_NAMES[selected]

    # --- Business threshold from OOF predictions --------------------------
    value_train = business.customer_value(X_train, bp)
    threshold = business.optimal_threshold(y_train, oof[selected], value_train, bp)
    f1_thr = E.best_f1_threshold(y_train, oof[selected])

    # --- Hold-out evaluation ----------------------------------------------
    test_rows = []
    for key, p in test_pred.items():
        m = E.metrics_at(y_test, p, threshold if key == selected else 0.5)
        lo, hi = E.bootstrap_ci(y_test, p)
        plo, phi = E.bootstrap_ci(y_test, p, average_precision_score)
        m.update({"key": key, "model": DISPLAY_NAMES[key], "auc_ci_low": lo, "auc_ci_high": hi,
                  "pr_auc_ci_low": plo, "pr_auc_ci_high": phi})
        test_rows.append(m)
    test = pd.DataFrame(test_rows)
    delong = pd.DataFrame(
        [{"model": DISPLAY_NAMES[k], **E.delong_test(y_test, test_pred[selected], p)}
         for k, p in test_pred.items() if k != selected]
    ).rename(columns={"auc_1": f"auc_{selected}", "auc_2": "auc_other"})

    # --- Inference latency --------------------------------------------------
    one = X_test.iloc[[0]]
    t0 = time.perf_counter()
    for _ in range(20):
        fitted[selected].predict_proba(one)
    latency_ms = (time.perf_counter() - t0) / 20 * 1000

    imb = imbalance_experiment(best_params["lightgbm"], X_train, y_train, splits)

    # --- Persist ------------------------------------------------------------
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    cv.to_csv(TABLES_DIR / "model_cv_results.csv", index=False)
    test.to_csv(TABLES_DIR / "model_test_results.csv", index=False)
    delong.to_csv(TABLES_DIR / "model_delong_tests.csv", index=False)
    imb.to_csv(TABLES_DIR / "imbalance_experiment.csv", index=False)
    if tuning_log:  # a --reuse run keeps the log of the run that actually tuned
        pd.DataFrame(tuning_log).to_csv(TABLES_DIR / "tuning_log.csv", index=False)
    elif (TABLES_DIR / "tuning_log.csv").exists():
        tuning_log = pd.read_csv(TABLES_DIR / "tuning_log.csv").to_dict("records")
    pd.DataFrame({"customerID": X_train["customerID"].values, "y": y_train.values,
                  **{DISPLAY_NAMES[k]: v for k, v in oof.items()}}).to_csv(TABLES_DIR / "oof_predictions.csv", index=False)
    pd.DataFrame({"customerID": X_test["customerID"].values, "y": y_test.values,
                  **{DISPLAY_NAMES[k]: v for k, v in test_pred.items()}}).to_csv(TABLES_DIR / "test_predictions.csv", index=False)
    joblib.dump(fitted[selected], MODELS_DIR / "churn_model.joblib", compress=3)
    joblib.dump(fitted, MODELS_DIR / "all_models.joblib", compress=3)

    # --- Figures ------------------------------------------------------------
    named_test = {DISPLAY_NAMES[k]: v for k, v in test_pred.items()}
    E.fig_model_comparison(cv, test, sel_name)
    E.fig_roc_pr(y_test, named_test, {sel_name, "Logistic Regression"})
    calib = [sel_name] + [m for m in ("Logistic Regression", "TabularNet (DL)") if m != sel_name]
    E.fig_calibration(y_test, named_test, calib[:3])
    E.fig_confusion(y_test, test_pred[selected], threshold, sel_name)
    deciles = E.fig_gains_lift(y_test, test_pred[selected], sel_name)
    E.fig_threshold_tradeoff(y_test, test_pred[selected], threshold, f1_thr)
    net = fitted["tabular_net"].named_steps["model"]
    E.fig_learning_curve(net.history_, net.best_epoch_)

    return {
        "selected_key": selected,
        "selected_model": sel_name,
        "threshold": threshold,
        "f1_threshold": f1_thr,
        "cv": cv,
        "test": test,
        "delong": delong,
        "imbalance": imb,
        "deciles": deciles,
        "best_params": best_params,
        "oof": oof,
        "test_pred": test_pred,
        "fitted": fitted,
        "latency_ms": latency_ms,
        "stack_weights": {DISPLAY_NAMES[k]: float(v) for k, v in stack.weights.items()},
        "soft_voting_members": [DISPLAY_NAMES[k] for k in top3],
        "within_1se": within_1se,
        "cv_standard_error": float(se),
        "tuning_log": tuning_log,
        "splits": splits,
    }
