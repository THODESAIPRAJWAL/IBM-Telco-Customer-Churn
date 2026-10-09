"""End-to-end pipeline: data -> EDA -> survival -> modelling -> business -> XAI -> segments -> fairness -> report.

Usage
-----
    python run_pipeline.py            # full run (Optuna tuning, ~20-30 min on a laptop CPU)
    python run_pipeline.py --fast     # smoke run with a handful of trials per model
    python run_pipeline.py --reuse    # skip tuning, reuse models/best_params.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from churn import business, data, eda, explain, fairness, report, segmentation, survival, train  # noqa: E402
from churn.config import (  # noqa: E402
    CATEGORY_LEVELS,
    MODELS_DIR,
    REPORTS_DIR,
    ROOT,
    TABLES_DIR,
    TARGET,
    business_params,
    ensure_dirs,
    load_config,
)
from churn.models import DISPLAY_NAMES  # noqa: E402
from churn.predict import MODEL_FEATURES  # noqa: E402

log = logging.getLogger("pipeline")


def _versions() -> dict[str, str]:
    import catboost, lightgbm, optuna, shap, sklearn, torch, xgboost  # noqa: E401

    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__, "catboost": catboost.__version__, "torch": torch.__version__,
            "shap": shap.__version__, "optuna": optuna.__version__, "pandas": pd.__version__, "numpy": np.__version__}


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return o.to_dict("records")
    return str(o)


def main(fast: bool = False, reuse: bool = False) -> dict:
    ensure_dirs()
    t_start = time.perf_counter()
    bp = business_params()
    timings = {}

    def step(name):
        timings[name] = time.perf_counter()
        log.info("=== %s ===", name)

    # 1. Data ---------------------------------------------------------------
    step("data")
    raw = data.load_raw()
    data.data_quality_report(raw).to_csv(TABLES_DIR / "data_quality.csv", index=False)
    df = data.load_clean()
    df.to_parquet(ROOT / load_config()["data"]["processed_file"], index=False)
    X_train, X_test, y_train, y_test = data.split(df)
    data_hash = hashlib.sha256((ROOT / load_config()["data"]["raw_file"]).read_bytes()).hexdigest()[:16]

    # 2. EDA + statistics ----------------------------------------------------
    step("eda")
    eda_summary = eda.run(df)

    # 3. Survival analysis ---------------------------------------------------
    step("survival")
    surv_summary = survival.run(df)

    # 4. Modelling -------------------------------------------------------------
    step("modelling")
    res = train.run(X_train, X_test, y_train, y_test, fast=fast, reuse_params=reuse)
    sel, thr = res["selected_key"], res["threshold"]
    p_test = res["test_pred"][sel]
    p_oof = res["oof"][sel]
    model = res["fitted"][sel]

    # 5. Business case (test set, projected to the full base) ------------------
    step("business")
    scale = len(df) / len(X_test)
    v_test = business.customer_value(X_test, bp)
    policies = business.compare_policies(y_test, p_test, v_test, thr, bp)
    policies.to_csv(TABLES_DIR / "business_policies_test.csv")
    business.fig_profit_curve(y_test, p_test, v_test, bp, scale=scale)
    business.fig_policy_comparison(policies, scale=scale)
    grid = business.sensitivity_grid(y_test, p_test, v_test, bp)
    grid.to_csv(TABLES_DIR / "business_sensitivity.csv")
    business.fig_sensitivity(grid, bp, scale=scale)

    # 6. Explainability -------------------------------------------------------
    step("explainability")
    gbm = max(["xgboost", "lightgbm"], key=lambda k: res["cv"].set_index("key").loc[k, "cv_auc_mean"])
    xai = explain.run(model, DISPLAY_NAMES[sel], X_train, X_test, y_test, p_test, thr,
                      tree_pipe=res["fitted"][gbm], tree_name=DISPLAY_NAMES[gbm])

    # 7. Segmentation on honest (out-of-sample) scores for every customer -----
    step("segmentation")
    scores = pd.concat([pd.Series(p_oof, index=X_train.index), pd.Series(p_test, index=X_test.index)]).loc[df.index]
    seg = segmentation.run(df, scores.to_numpy(), business.customer_value(df, bp), thr)

    # 8. Fairness --------------------------------------------------------------
    step("fairness")
    targeted = business.ev_targets(p_test, v_test, bp)
    fair = fairness.run(X_test, y_test, p_test, targeted, res["best_params"]["lightgbm"], X_train, y_train, res["splits"])

    # 9. Persist metadata + report --------------------------------------------
    step("report")
    test_sel = res["test"].set_index("key").loc[sel]
    cv_sel = res["cv"].set_index("key").loc[sel]
    metadata = {
        "model_name": DISPLAY_NAMES[sel],
        "model_key": sel,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_sha256_16": data_hash,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": MODEL_FEATURES,
        "category_levels": {k: v for k, v in CATEGORY_LEVELS.items() if k in MODEL_FEATURES},
        "threshold": thr,
        "f1_threshold": res["f1_threshold"],
        "business_params": bp.__dict__,
        "cv_auc_mean": float(cv_sel["cv_auc_mean"]),
        "cv_auc_std": float(cv_sel["cv_auc_std"]),
        "test_metrics": {k: float(test_sel[k]) for k in
                         ["roc_auc", "auc_ci_low", "auc_ci_high", "pr_auc", "brier", "log_loss", "precision",
                          "recall", "f1", "specificity", "mcc", "lift_top10"]},
        "latency_ms_single": res["latency_ms"],
        "versions": _versions(),
    }
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2, default=_jsonable))

    summary = {
        "metadata": metadata,
        "eda": eda_summary,
        "survival": surv_summary,
        "modelling": {
            "cv": res["cv"].round(5).to_dict("records"),
            "test": res["test"].round(5).to_dict("records"),
            "delong": res["delong"].round(5).to_dict("records"),
            "imbalance": res["imbalance"].round(5).to_dict("records"),
            "deciles": res["deciles"].round(4).reset_index().to_dict("records"),
            "best_params": res["best_params"],
            "stack_weights": res["stack_weights"],
            "soft_voting_members": res["soft_voting_members"],
            "within_1se": res["within_1se"],
            "cv_standard_error": res["cv_standard_error"],
            "tuning_log": res["tuning_log"],
        },
        "business": {
            "scale_to_full_base": scale,
            "policies_test": policies.reset_index().to_dict("records"),
            "sensitivity": grid.reset_index().to_dict("records"),
        },
        "explainability": xai,
        "segmentation": seg,
        "fairness": fair,
        "runtime_minutes": (time.perf_counter() - t_start) / 60,
        "fast_mode": fast,
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2, default=_jsonable))
    report.write(summary)
    log.info("Done in %.1f min. Selected: %s (test AUC %.4f)", summary["runtime_minutes"],
             DISPLAY_NAMES[sel], metadata["test_metrics"]["roc_auc"])
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="few Optuna trials per model (smoke test)")
    ap.add_argument("--reuse", action="store_true", help="reuse tuned hyper-parameters from models/best_params.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    main(fast=args.fast, reuse=args.reuse)
