"""REST API for churn scoring.

Run:  uvicorn api.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, model_validator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churn import __version__  # noqa: E402
from churn.predict import ChurnPredictor, risk_tier  # noqa: E402

YesNo = Literal["Yes", "No"]
NoInternet = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "customerID": "7590-VHVEG", "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
        "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85,
        "TotalCharges": 29.85}})

    customerID: str | None = None
    gender: Literal["Female", "Male"] | None = None
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: NoInternet
    OnlineBackup: NoInternet
    DeviceProtection: NoInternet
    TechSupport: NoInternet
    StreamingTV: NoInternet
    StreamingMovies: NoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(gt=0, le=500)
    TotalCharges: float = Field(ge=0)

    @model_validator(mode="after")
    def _consistent(self):
        addons = [self.OnlineSecurity, self.OnlineBackup, self.DeviceProtection, self.TechSupport,
                  self.StreamingTV, self.StreamingMovies]
        if (self.InternetService == "No") != all(a == "No internet service" for a in addons):
            raise ValueError("'No internet service' add-ons must match InternetService == 'No'")
        if (self.PhoneService == "No") != (self.MultipleLines == "No phone service"):
            raise ValueError("MultipleLines == 'No phone service' must match PhoneService == 'No'")
        return self


class Contribution(BaseModel):
    feature: str
    value: str
    contribution: float


class Action(BaseModel):
    action: str
    new_probability: float
    risk_reduction: float


class Prediction(BaseModel):
    customerID: str | None
    churn_probability: float
    risk_tier: str
    value_at_risk: float
    expected_campaign_value: float
    target_for_retention: bool
    explanation: list[Contribution] | None = None
    next_best_actions: list[Action] | None = None


STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["predictor"] = ChurnPredictor.load()
    yield
    STATE.clear()


app = FastAPI(title="Telco Churn Scoring API", version=__version__, lifespan=lifespan,
              description="Churn probability, risk tier, value-aware retention decision, SHAP explanation "
                          "and simulated next-best actions for telco customers.")


def _predictor() -> ChurnPredictor:
    if "predictor" not in STATE:
        raise HTTPException(503, "model not loaded")
    return STATE["predictor"]


@app.get("/health")
def health():
    p = _predictor()
    return {"status": "ok", "model": p.metadata["model_name"], "trained_at": p.metadata["trained_at"],
            "api_version": __version__}


@app.get("/model")
def model_info():
    md = _predictor().metadata
    return {k: md[k] for k in ("model_name", "trained_at", "threshold", "business_params", "cv_auc_mean",
                               "test_metrics", "features")}


def _to_frame(customers: list[Customer]) -> pd.DataFrame:
    return pd.DataFrame([c.model_dump() for c in customers])


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer, explain: bool = Query(False, description="include SHAP explanation (~1s)"),
            actions: bool = Query(False, description="include simulated next-best actions")):
    p = _predictor()
    df = _to_frame([customer])
    s = p.score(df).iloc[0]
    out = Prediction(customerID=customer.customerID, churn_probability=float(s["churn_probability"]),
                     risk_tier=risk_tier(float(s["churn_probability"]))[0], value_at_risk=float(s["value_at_risk"]),
                     expected_campaign_value=float(s["expected_campaign_value"]),
                     target_for_retention=bool(s["target_for_retention"]))
    if explain:
        ex = p.explain(df).head(8)
        out.explanation = [Contribution(feature=r.feature, value=str(r.value), contribution=float(r.contribution))
                           for r in ex.itertuples()]
    if actions:
        nba = p.next_best_actions(df)
        out.next_best_actions = [Action(action=r.action, new_probability=float(r.new_probability),
                                        risk_reduction=float(r.risk_reduction)) for r in nba.itertuples()]
    return out


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(customers: list[Customer]):
    if not customers:
        return []
    if len(customers) > 10_000:
        raise HTTPException(413, "batch limited to 10,000 customers")
    p = _predictor()
    scored = p.score(_to_frame(customers))
    return [Prediction(customerID=c.customerID, churn_probability=float(r["churn_probability"]),
                       risk_tier=r["risk_tier"], value_at_risk=float(r["value_at_risk"]),
                       expected_campaign_value=float(r["expected_campaign_value"]),
                       target_for_retention=bool(r["target_for_retention"]))
            for c, (_, r) in zip(customers, scored.iterrows())]
