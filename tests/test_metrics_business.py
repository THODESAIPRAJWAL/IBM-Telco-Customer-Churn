import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from churn import business
from churn import evaluate as E
from churn.config import BusinessParams


@pytest.fixture
def scores():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 800)
    p1 = np.clip(0.3 * y + rng.normal(0.35, 0.2, 800), 0, 1)
    p2 = np.clip(0.2 * y + rng.normal(0.4, 0.25, 800), 0, 1)
    return y, p1, p2


def test_delong_matches_sklearn_auc(scores):
    y, p1, p2 = scores
    r = E.delong_test(y, p1, p2)
    assert np.isclose(r["auc_1"], roc_auc_score(y, p1))
    assert np.isclose(r["auc_2"], roc_auc_score(y, p2))
    assert r["p_value"] < 0.05  # p1 is built to be clearly better


def test_delong_identical_models(scores):
    y, p1, _ = scores
    assert E.delong_test(y, p1, p1)["p_value"] == 1.0


def test_bootstrap_ci_brackets_point_estimate(scores):
    y, p1, _ = scores
    lo, hi = E.bootstrap_ci(y, p1, n_boot=300)
    assert lo < roc_auc_score(y, p1) < hi


def test_lift_of_perfect_ranking():
    y = np.array([1] * 10 + [0] * 90)
    assert np.isclose(E.lift_at(y, y.astype(float), 0.1), 10.0)


def test_campaign_outcome_arithmetic():
    bp = BusinessParams(clv_horizon_months=10, gross_margin=0.5, contact_cost=5, incentive_cost=15,
                        offer_success_rate=0.5)
    y = np.array([1, 1, 0, 0])
    value = np.array([100.0, 200.0, 100.0, 100.0])
    out = business.campaign_outcome(y, np.array([True, False, True, False]), value, bp)
    assert out["n_targeted"] == 2 and out["churners_reached"] == 1
    assert out["campaign_cost"] == 40
    assert out["retained_value"] == 50  # 0.5 * 100
    assert out["net_profit"] == 10


def test_ev_rule_and_oracle_dominance():
    bp = BusinessParams()
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    p = np.clip(0.4 * y + rng.uniform(0, 0.6, 2000), 0, 1)
    value = rng.uniform(300, 1500, 2000)
    pol = business.compare_policies(y, p, value, 0.5, bp)
    oracle = pol.loc["Oracle (perfect foresight)", "net_profit"]
    assert (pol["net_profit"] <= oracle + 1e-9).all()
    assert pol.loc["No campaign", "net_profit"] == 0
    t = business.ev_targets(p, value, bp)
    assert np.array_equal(t, p * bp.offer_success_rate * value > bp.cost_per_target)


def test_customer_value():
    bp = BusinessParams(clv_horizon_months=24, gross_margin=0.6)
    v = business.customer_value(pd.DataFrame({"MonthlyCharges": [50.0]}), bp)
    assert np.isclose(v[0], 50 * 24 * 0.6)
