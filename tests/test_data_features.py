import numpy as np
import pandas as pd

from churn import data
from churn.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, FeatureEngineer


def test_clean_fixes_blank_total_charges():
    raw = data.load_raw()
    assert raw["TotalCharges"].astype(str).str.strip().eq("").sum() == 11
    df = data.clean(raw)
    assert df["TotalCharges"].isna().sum() == 0
    assert (df.loc[df["tenure"] == 0, "TotalCharges"] == 0).all()
    assert set(df["Churn"].unique()) == {0, 1}


def test_dataset_shape_and_rate(df):
    assert df.shape == (7043, 21)
    assert abs(df["Churn"].mean() - 0.2654) < 1e-3
    assert df["customerID"].is_unique


def test_validate_flags_bad_levels(sample):
    bad = sample.copy()
    bad.loc[0, "Contract"] = "Three year"
    bad.loc[1, "tenure"] = -5
    problems = data.validate(bad)
    assert any("Contract" in p for p in problems)
    assert any("tenure" in p for p in problems)
    assert data.validate(sample) == []


def test_split_is_stratified_and_disjoint(df):
    X_tr, X_te, y_tr, y_te = data.split(df)
    assert len(X_te) == 1409
    assert set(X_tr["customerID"]).isdisjoint(X_te["customerID"])
    assert abs(y_tr.mean() - y_te.mean()) < 0.005


def test_feature_engineer_output(sample):
    out = FeatureEngineer().transform(sample)
    assert list(out.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert out[NUMERIC_FEATURES].isna().sum().sum() == 0
    assert "gender" not in out.columns and "customerID" not in out.columns
    assert out["num_protection"].between(0, 4).all()
    assert out["num_services"].between(0, 8).all()


def test_feature_engineer_is_stateless(df, sample):
    """Row-wise transforms: the output for a row can't depend on other rows (no leakage)."""
    fe = FeatureEngineer().fit(df)
    a = fe.transform(sample)
    b = pd.concat([FeatureEngineer().transform(sample.iloc[[i]]) for i in range(20)])
    pd.testing.assert_frame_equal(a.iloc[:20].reset_index(drop=True), b.reset_index(drop=True))


def test_tenure_zero_customer():
    row = data.load_clean().query("tenure == 0").iloc[[0]]
    out = FeatureEngineer().transform(row)
    assert np.isclose(out["avg_monthly_charge"].iloc[0], row["MonthlyCharges"].iloc[0])
    assert out["new_customer"].iloc[0] == 1
