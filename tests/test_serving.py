"""Tests for the trained artefact, the predictor and the REST API (skipped until the pipeline has run)."""

import numpy as np
import pytest

from conftest import needs_artifacts

pytestmark = needs_artifacts

EXAMPLE = {
    "customerID": "TEST-0001", "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 99.5, "TotalCharges": 190.0,
}


@pytest.fixture(scope="module")
def predictor():
    from churn.predict import ChurnPredictor

    return ChurnPredictor.load()


def test_scores_are_probabilities(predictor, sample):
    s = predictor.score(sample)
    assert s["churn_probability"].between(0, 1).all()
    assert set(s["risk_tier"]) <= {"Low", "Medium", "High", "Critical"}


def test_known_risk_ordering(predictor):
    import pandas as pd

    risky = pd.DataFrame([EXAMPLE])
    safe = risky.assign(Contract="Two year", tenure=60, PaymentMethod="Credit card (automatic)",
                        OnlineSecurity="Yes", TechSupport="Yes", TotalCharges=5900.0)
    assert predictor.predict_proba(risky)[0] > 0.5 > predictor.predict_proba(safe)[0]


def test_shap_explanation_is_additive(predictor):
    import pandas as pd

    row = pd.DataFrame([EXAMPLE])
    ex = predictor.explain(row)
    p = predictor.predict_proba(row)[0]
    assert np.isclose(ex["base_value"].iloc[0] + ex["contribution"].sum(), p, atol=0.02)
    assert ex.iloc[0]["feature"] in {"Contract", "tenure", "InternetService", "PaymentMethod", "MonthlyCharges",
                                     "TotalCharges", "OnlineSecurity", "TechSupport"}


def test_next_best_actions(predictor):
    import pandas as pd

    nba = predictor.next_best_actions(pd.DataFrame([EXAMPLE]))
    assert len(nba) >= 3
    assert nba["risk_reduction"].is_monotonic_decreasing


def test_api_endpoints():
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        r = client.post("/predict?actions=true", json=EXAMPLE)
        assert r.status_code == 200
        body = r.json()
        assert 0 <= body["churn_probability"] <= 1 and body["next_best_actions"]
        bad = dict(EXAMPLE, InternetService="No")  # add-ons inconsistent with no internet
        assert client.post("/predict", json=bad).status_code == 422
        assert client.post("/predict", json=dict(EXAMPLE, Contract="Forever")).status_code == 422
        batch = client.post("/predict/batch", json=[EXAMPLE, dict(EXAMPLE, customerID="TEST-0002", tenure=40)])
        assert batch.status_code == 200 and len(batch.json()) == 2
