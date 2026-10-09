"""Shared, cached data access and chart helpers for the dashboard pages."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churn import business as B  # noqa: E402
from churn.config import CATEGORY_LEVELS, BusinessParams  # noqa: E402
from churn.predict import MODEL_FEATURES, ChurnPredictor, risk_tier  # noqa: E402

TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"

# Visual system (matches the static report): colour follows the entity
RETAINED, CHURNED = "#2a78d6", "#eb6834"
GHOST = "#c9c8c1"
MODEL_COLORS = {
    "Logistic Regression": "#2a78d6",
    "Stacked Ensemble": "#eb6834",
    "Random Forest": "#1baf7a",
    "XGBoost": "#eda100",
    "LightGBM": "#e87ba4",
    "CatBoost": "#008300",
    "TabularNet (DL)": "#4a3aa7",
    "Soft-Voting Ensemble": "#e34948",
}
CHURN_SCALE = alt.Scale(domain=["Retained", "Churned"], range=[RETAINED, CHURNED])
TIER_BADGE = {"Low": ("green", ":material/check_circle:"), "Medium": ("orange", ":material/info:"),
              "High": ("orange", ":material/warning:"), "Critical": ("red", ":material/error:")}


def artifacts_ready() -> bool:
    return (ROOT / "models" / "churn_model.joblib").exists() and (ROOT / "reports" / "metrics.json").exists()


def require_artifacts() -> None:
    if not artifacts_ready():
        st.error("Model artefacts not found. Run `python run_pipeline.py` from the project root first.",
                 icon=":material/error:")
        st.stop()


@st.cache_resource(show_spinner="Loading model...")
def predictor() -> ChurnPredictor:
    return ChurnPredictor.load(ROOT / "models")


@st.cache_data
def metrics() -> dict:
    return json.loads((ROOT / "reports" / "metrics.json").read_text())


@st.cache_data
def table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / f"{name}.csv")


@st.cache_data
def customers() -> pd.DataFrame:
    processed = ROOT / "data" / "processed" / "telco_clean.parquet"
    if processed.exists():
        df = pd.read_parquet(processed)
    else:  # fresh clone: rebuild from the raw CSV (downloaded on demand)
        from churn.data import load_clean

        df = load_clean()
    df["Churn label"] = np.where(df["Churn"] == 1, "Churned", "Retained")
    return df


@st.cache_data
def scored_customers() -> pd.DataFrame:
    """All customers with out-of-sample scores, segment and persona."""
    s = table("customers_scored")
    raw = customers().drop(columns=["Churn", "MonthlyCharges", "tenure", "Contract"])
    return s.merge(raw, on="customerID")


def business_from_metadata() -> BusinessParams:
    return BusinessParams(**metrics()["metadata"]["business_params"])


def selected_model_name() -> str:
    return metrics()["metadata"]["model_name"]


def money(x: float) -> str:
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def rate_bar(df: pd.DataFrame, col: str, overall: float, height: int = 220) -> alt.LayerChart:
    """Churn rate per level (single series -> one colour) with the overall-rate rule."""
    g = df.groupby(col, observed=True)["Churn"].agg(churn_rate="mean", customers="size").reset_index()
    g[col] = g[col].astype(str)
    bars = alt.Chart(g).mark_bar(color=CHURNED, cornerRadiusEnd=4, height={"band": 0.65}).encode(
        y=alt.Y(f"{col}:N", sort="-x", title=None),
        x=alt.X("churn_rate:Q", title="Churn rate", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 0.75])),
        tooltip=[alt.Tooltip(f"{col}:N"), alt.Tooltip("churn_rate:Q", format=".1%", title="Churn rate"),
                 alt.Tooltip("customers:Q", format=",", title="Customers")],
    )
    labels = bars.mark_text(align="left", dx=4, fontSize=11).encode(text=alt.Text("churn_rate:Q", format=".0%"))
    rule = alt.Chart(pd.DataFrame({"x": [overall]})).mark_rule(strokeWidth=1).encode(x="x:Q")
    return (bars + labels + rule).properties(height=height)
