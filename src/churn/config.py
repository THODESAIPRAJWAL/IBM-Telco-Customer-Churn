"""Project paths, constants and configuration loading."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = ROOT / "config.yaml"

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

TARGET = "Churn"
ID_COL = "customerID"

# Raw input schema (what a customer record looks like before feature engineering)
NUMERIC_RAW = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_RAW = ["SeniorCitizen"]
CATEGORICAL_RAW = [
    "gender",
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
]
RAW_FEATURES = ["gender", "SeniorCitizen", *CATEGORICAL_RAW[1:], *NUMERIC_RAW]

# `gender` is excluded from modelling: it carries no signal (chi-square p ~ 0.49)
# and is a protected attribute. The fairness audit shows dropping it costs nothing.
EXCLUDED_FEATURES = ["gender"]

CATEGORY_LEVELS: dict[str, list[str]] = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}


@dataclass(frozen=True)
class BusinessParams:
    clv_horizon_months: int = 24
    gross_margin: float = 0.60
    contact_cost: float = 10.0
    incentive_cost: float = 60.0
    offer_success_rate: float = 0.35

    @property
    def cost_per_target(self) -> float:
        return self.contact_cost + self.incentive_cost


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    with open(CONFIG_FILE, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def business_params() -> BusinessParams:
    return BusinessParams(**load_config()["business"])


def ensure_dirs() -> None:
    for d in (DATA_DIR / "raw", DATA_DIR / "processed", MODELS_DIR, FIGURES_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)
