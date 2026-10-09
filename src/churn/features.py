"""Feature engineering.

All transforms here are *row-wise and stateless* (no statistics are learned from
the data), so they cannot leak information across the train/test boundary and
are safe to run identically at training and inference time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import EXCLUDED_FEATURES, ID_COL, TARGET

ADDON_SERVICES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
PROTECTION_SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
STREAMING_SERVICES = ["StreamingTV", "StreamingMovies"]

CATEGORICAL_FEATURES = [
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_bucket",
]
NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "log_tenure",
    "num_services",
    "num_protection",
    "num_streaming",
    "avg_monthly_charge",
    "charge_delta",
    "monthly_per_service",
    "contract_months",
    "has_internet",
    "is_fiber",
    "auto_payment",
    "electronic_check",
    "has_family",
    "fiber_no_support",
    "new_customer",
]

TENURE_BINS = [-1, 6, 12, 24, 48, 72]
TENURE_LABELS = ["0-6m", "7-12m", "13-24m", "25-48m", "49-72m"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Domain features for telco churn.

    * **Engagement** - number of add-on / protection / streaming services.
    * **Pricing** - average historical charge vs current charge (`charge_delta`
      flags customers whose bill has recently gone up), price per service.
    * **Commitment** - contract length in months, automatic payment, tenure bucket.
    * **Known risk pattern** - fibre customers without security/tech support.
    """

    def __init__(self, drop: tuple[str, ...] = tuple(EXCLUDED_FEATURES)):
        self.drop = drop

    def fit(self, X: pd.DataFrame, y=None):  # noqa: D401 - sklearn API
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.drop(columns=[c for c in (ID_COL, TARGET, *self.drop) if c in X.columns]).copy()

        yes = lambda cols: (df[cols] == "Yes").sum(axis=1)  # noqa: E731
        df["num_protection"] = yes(PROTECTION_SERVICES)
        df["num_streaming"] = yes(STREAMING_SERVICES)
        df["num_services"] = yes(ADDON_SERVICES) + (df["PhoneService"] == "Yes") + (
            df["MultipleLines"] == "Yes"
        )

        tenure = df["tenure"].astype(float)
        df["log_tenure"] = np.log1p(tenure)
        df["avg_monthly_charge"] = np.where(
            tenure > 0, df["TotalCharges"] / tenure.replace(0, np.nan), df["MonthlyCharges"]
        )
        df["charge_delta"] = df["MonthlyCharges"] - df["avg_monthly_charge"]
        df["monthly_per_service"] = df["MonthlyCharges"] / (df["num_services"] + 1)

        df["contract_months"] = df["Contract"].map(
            {"Month-to-month": 1, "One year": 12, "Two year": 24}
        )
        df["has_internet"] = (df["InternetService"] != "No").astype(int)
        df["is_fiber"] = (df["InternetService"] == "Fiber optic").astype(int)
        df["auto_payment"] = df["PaymentMethod"].str.contains("automatic").astype(int)
        df["electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)
        df["has_family"] = ((df["Partner"] == "Yes") | (df["Dependents"] == "Yes")).astype(int)
        df["fiber_no_support"] = (
            (df["InternetService"] == "Fiber optic")
            & (df["OnlineSecurity"] != "Yes")
            & (df["TechSupport"] != "Yes")
        ).astype(int)
        df["new_customer"] = (tenure <= 6).astype(int)
        df["tenure_bucket"] = pd.cut(tenure, bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)

        return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    def get_feature_names_out(self, input_features=None):
        return np.array(NUMERIC_FEATURES + CATEGORICAL_FEATURES, dtype=object)


def make_preprocessor(scale: bool = True) -> ColumnTransformer:
    """One-hot categoricals (+ optional scaling) for sklearn / XGBoost / LightGBM."""
    return ColumnTransformer(
        [
            ("num", StandardScaler() if scale else "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )


def engineered_frame(X: pd.DataFrame) -> pd.DataFrame:
    return FeatureEngineer().transform(X)
