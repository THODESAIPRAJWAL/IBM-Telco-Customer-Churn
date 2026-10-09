"""Data acquisition, cleaning and validation."""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    CATEGORY_LEVELS,
    ID_COL,
    NUMERIC_RAW,
    RAW_FEATURES,
    ROOT,
    TARGET,
    load_config,
)

log = logging.getLogger(__name__)


def download_data(force: bool = False) -> Path:
    """Download the IBM Telco churn CSV (idempotent)."""
    cfg = load_config()["data"]
    path = ROOT / cfg["raw_file"]
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading dataset from %s", cfg["url"])
    urllib.request.urlretrieve(cfg["url"], path)
    return path


def load_raw() -> pd.DataFrame:
    return pd.read_csv(download_data())


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix types and the known data-quality issues.

    * ``TotalCharges`` is stored as text; 11 customers with ``tenure == 0`` have a
      blank value. They have not been billed yet, so the correct value is 0.
    * ``Churn`` is mapped to 1/0.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(r"^\s*$", np.nan, regex=True))
    blank = df["TotalCharges"].isna()
    if blank.any():
        assert (df.loc[blank, "tenure"] == 0).all(), "blank TotalCharges outside tenure==0"
        df.loc[blank, "TotalCharges"] = 0.0
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    if TARGET in df and not pd.api.types.is_numeric_dtype(df[TARGET]):
        df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)
    return df


def validate(df: pd.DataFrame, require_target: bool = True) -> list[str]:
    """Lightweight schema checks. Returns a list of human-readable problems."""
    problems: list[str] = []
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
        return problems
    if require_target and TARGET not in df.columns:
        problems.append(f"missing target column '{TARGET}'")
    for col, levels in CATEGORY_LEVELS.items():
        bad = set(df[col].dropna().unique()) - set(levels)
        if bad:
            problems.append(f"{col}: unexpected levels {sorted(bad)}")
    for col in NUMERIC_RAW:
        if df[col].isna().any():
            problems.append(f"{col}: {int(df[col].isna().sum())} missing values")
        if (df[col] < 0).any():
            problems.append(f"{col}: negative values")
    if not set(df["SeniorCitizen"].unique()) <= {0, 1}:
        problems.append("SeniorCitizen must be 0/1")
    return problems


def load_clean() -> pd.DataFrame:
    df = clean(load_raw())
    problems = validate(df)
    if problems:
        raise ValueError("Data validation failed: " + "; ".join(problems))
    return df


def data_quality_report(raw: pd.DataFrame) -> pd.DataFrame:
    """Column-level profile of the raw file (types, missing, blanks, cardinality)."""
    rows = []
    for col in raw.columns:
        s = raw[col]
        blanks = int(s.astype(str).str.strip().eq("").sum())
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "missing": int(s.isna().sum()),
                "blank_strings": blanks,
                "unique": int(s.nunique()),
                "example": s.iloc[0],
            }
        )
    return pd.DataFrame(rows)


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified hold-out split. The test set is touched only for final reporting."""
    cfg = load_config()["split"]
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=cfg["test_size"], stratify=y, random_state=cfg["random_state"]
    )


def features_only(X: pd.DataFrame) -> pd.DataFrame:
    """Drop identifiers so only model inputs remain."""
    return X.drop(columns=[c for c in (ID_COL, TARGET) if c in X.columns])
