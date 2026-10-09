"""Inference: scoring, risk tiers, explanations and next-best-action simulation.

This module is intentionally light (no plotting) so the API and dashboard can
import it cheaply.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import CATEGORY_LEVELS, EXCLUDED_FEATURES, MODELS_DIR, RAW_FEATURES, BusinessParams

MODEL_FEATURES = [f for f in RAW_FEATURES if f not in EXCLUDED_FEATURES]

RISK_TIERS = [  # (upper bound, label, status token)
    (0.20, "Low", "good"),
    (0.40, "Medium", "warning"),
    (0.60, "High", "serious"),
    (1.01, "Critical", "critical"),
]


def risk_tier(p: float) -> tuple[str, str]:
    for ub, label, status in RISK_TIERS:
        if p < ub:
            return label, status
    return "Critical", "critical"


class RawSpaceModel:
    """Numeric facade over the raw customer schema for model-agnostic SHAP.

    Categoricals are exposed as integer codes so SHAP maskers can shuffle them;
    the facade decodes back to strings before calling the real pipeline, so
    attributions are expressed in the business's own vocabulary
    (e.g. ``Contract = Month-to-month``), not in one-hot columns.
    """

    def __init__(self, model, features: list[str] = MODEL_FEATURES):
        self.model = model
        self.features = list(features)

    def encode(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        for f in self.features:
            if f in CATEGORY_LEVELS:
                cols.append(df[f].map({v: i for i, v in enumerate(CATEGORY_LEVELS[f])}).to_numpy(dtype=float))
            else:
                cols.append(df[f].to_numpy(dtype=float))
        return np.column_stack(cols)

    def decode(self, arr: np.ndarray) -> pd.DataFrame:
        arr = np.atleast_2d(arr)
        data = {}
        for j, f in enumerate(self.features):
            if f in CATEGORY_LEVELS:
                levels = np.array(CATEGORY_LEVELS[f], dtype=object)
                data[f] = levels[np.clip(np.rint(arr[:, j]).astype(int), 0, len(levels) - 1)]
            elif f == "SeniorCitizen":
                data[f] = np.rint(arr[:, j]).astype(int)
            else:
                data[f] = arr[:, j]
        return pd.DataFrame(data)

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.decode(arr))[:, 1]


# Candidate retention actions: (label, applicability test, edit)
ACTIONS = [
    ("Upgrade to a 1-year contract (with loyalty discount)",
     lambda r: r["Contract"] == "Month-to-month", {"Contract": "One year"}),
    ("Upgrade to a 2-year contract (price lock)",
     lambda r: r["Contract"] != "Two year", {"Contract": "Two year"}),
    ("Bundle Online Security free for 6 months",
     lambda r: r["OnlineSecurity"] == "No", {"OnlineSecurity": "Yes"}),
    ("Bundle Tech Support free for 6 months",
     lambda r: r["TechSupport"] == "No", {"TechSupport": "Yes"}),
    ("Move to automatic card payment (bill credit)",
     lambda r: r["PaymentMethod"] in ("Electronic check", "Mailed check"), {"PaymentMethod": "Credit card (automatic)"}),
    ("15% loyalty discount on monthly bill",
     lambda r: True, {"MonthlyCharges": lambda r: round(r["MonthlyCharges"] * 0.85, 2)}),
]


@dataclass
class ChurnPredictor:
    model: Any
    metadata: dict
    background: pd.DataFrame | None = None

    @classmethod
    def load(cls, models_dir: Path = MODELS_DIR) -> "ChurnPredictor":
        model = joblib.load(models_dir / "churn_model.joblib")
        metadata = json.loads((models_dir / "model_metadata.json").read_text())
        bg_file = models_dir / "shap_background.csv"
        background = pd.read_csv(bg_file) if bg_file.exists() else None
        return cls(model, metadata, background)

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"])

    @property
    def business(self) -> BusinessParams:
        return BusinessParams(**self.metadata["business_params"])

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(df[MODEL_FEATURES])[:, 1]

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Probability, risk tier, value at risk and the recommended decision."""
        bp = self.business
        p = self.predict_proba(df)
        value = df["MonthlyCharges"].to_numpy(float) * bp.clv_horizon_months * bp.gross_margin
        ev = p * bp.offer_success_rate * value - bp.cost_per_target
        tiers = [risk_tier(x)[0] for x in p]
        out = pd.DataFrame({
            "churn_probability": p.round(4),
            "risk_tier": tiers,
            "value_at_risk": value.round(2),
            "expected_campaign_value": ev.round(2),
            "target_for_retention": ev > 0,
            "above_uniform_threshold": p >= self.threshold,
        }, index=df.index)
        if "customerID" in df:
            out.insert(0, "customerID", df["customerID"].values)
        return out

    @cached_property
    def _explainer(self):
        import shap

        if self.background is None:
            raise RuntimeError("SHAP background sample not found - run the training pipeline first.")
        facade = RawSpaceModel(self.model)
        bg = facade.encode(self.background[MODEL_FEATURES])
        return facade, shap.PermutationExplainer(facade, shap.maskers.Independent(bg, max_samples=len(bg)))

    def explain(self, row: pd.DataFrame, max_evals: int = 200) -> pd.DataFrame:
        """SHAP contributions (in probability points) for a single customer."""
        facade, explainer = self._explainer
        sv = explainer(facade.encode(row[MODEL_FEATURES].iloc[[0]]), max_evals=max_evals, silent=True)
        values = sv.values[0]
        return (
            pd.DataFrame({
                "feature": MODEL_FEATURES,
                "value": [row[f].iloc[0] for f in MODEL_FEATURES],
                "contribution": values,
            })
            .assign(abs_contribution=lambda d: d["contribution"].abs())
            .sort_values("abs_contribution", ascending=False)
            .drop(columns="abs_contribution")
            .reset_index(drop=True)
            .pipe(lambda d: d.assign(base_value=float(sv.base_values[0])))
        )

    def next_best_actions(self, row: pd.DataFrame) -> pd.DataFrame:
        """Simulate each applicable retention action and rank by risk reduction.

        NOTE: these are *model-implied* (associational) effects. They prioritise
        which offers to test; causal lift must be confirmed with an A/B test or
        an uplift model before rollout.
        """
        base_row = row[MODEL_FEATURES].iloc[[0]].copy()
        r = base_row.iloc[0]
        base_p = float(self.predict_proba(base_row)[0])
        bp = self.business
        rows = []
        for label, applies, edit in ACTIONS:
            if r["InternetService"] == "No" and ("Security" in label or "Tech Support" in label):
                continue
            if not applies(r):
                continue
            cf = base_row.copy()
            for k, v in edit.items():
                cf[k] = v(r) if callable(v) else v
            p = float(self.predict_proba(cf)[0])
            value = float(cf["MonthlyCharges"].iloc[0]) * bp.clv_horizon_months * bp.gross_margin
            rows.append({"action": label, "new_probability": p, "risk_reduction": base_p - p,
                         "new_value_at_risk": value})
        return pd.DataFrame(rows).sort_values("risk_reduction", ascending=False).reset_index(drop=True)
