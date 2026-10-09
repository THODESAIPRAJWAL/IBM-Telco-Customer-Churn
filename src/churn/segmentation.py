"""Customer segmentation: an actionable risk x value matrix and behavioural personas."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from . import plotting as P
from .config import TABLES_DIR, TARGET
from .features import FeatureEngineer

plt = P.plt

QUADRANTS = {
    (True, True): "Priority save",
    (True, False): "Low-cost save",
    (False, True): "Protect & grow",
    (False, False): "Maintain",
}
PLAYBOOK = {
    "Priority save": "Proactive outreach by a retention specialist; contract upgrade with loyalty pricing",
    "Low-cost save": "Automated digital offer (auto-pay credit, free add-on trial); no costly incentives",
    "Protect & grow": "Loyalty perks and upsell of protection bundles; avoid unnecessary discounts",
    "Maintain": "Business as usual; monitor for risk changes",
}


def risk_value_matrix(df: pd.DataFrame, p: np.ndarray, value: np.ndarray, risk_cut: float) -> pd.DataFrame:
    d = df[["customerID", TARGET, "MonthlyCharges", "tenure", "Contract"]].copy()
    d["churn_probability"] = p
    d["value_at_risk"] = value
    d["expected_loss"] = p * value
    value_cut = float(np.median(value))
    d["segment"] = [QUADRANTS[(r >= risk_cut, v >= value_cut)] for r, v in zip(p, value)]
    return d


def segment_summary(d: pd.DataFrame) -> pd.DataFrame:
    s = d.groupby("segment").agg(
        customers=("customerID", "size"),
        actual_churn_rate=(TARGET, "mean"),
        mean_predicted_risk=("churn_probability", "mean"),
        avg_monthly_charges=("MonthlyCharges", "mean"),
        avg_tenure=("tenure", "mean"),
        expected_margin_loss=("expected_loss", "sum"),
    )
    s["share_of_expected_loss"] = s["expected_margin_loss"] / s["expected_margin_loss"].sum()
    s["playbook"] = s.index.map(PLAYBOOK)
    return s.reindex(["Priority save", "Low-cost save", "Protect & grow", "Maintain"])


def fig_risk_value(d: pd.DataFrame, summary: pd.DataFrame, risk_cut: float) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.4), layout="constrained")
    for label, name in [(0, "Stayed"), (1, "Churned")]:
        g = d[d[TARGET] == label]
        ax.scatter(g["churn_probability"], g["value_at_risk"], s=8, alpha=0.35 if label else 0.25,
                   color=P.CHURN_COLORS[label], label=f"Actually {name.lower()}", linewidths=0)
    vcut = float(np.median(d["value_at_risk"]))
    ax.axvline(risk_cut, color=P.INK_2, lw=1)
    ax.axhline(vcut, color=P.INK_2, lw=1)
    pos = {"Priority save": (0.99, 0.98, "right", "top"),
           "Low-cost save": (0.99, 0.03, "right", "bottom"),
           "Protect & grow": (0.01, 0.98, "left", "top"),
           "Maintain": (0.01, 0.03, "left", "bottom")}
    for seg, (x, y, ha, va) in pos.items():
        r = summary.loc[seg]
        ax.text(x, y, f"{seg}\n{int(r['customers']):,} customers | churn {P.pct(r['actual_churn_rate'], 0)}\n"
                      f"{P.pct(r['share_of_expected_loss'], 0)} of expected loss",
                ha=ha, va=va, fontsize=9, color=P.INK, transform=ax.transAxes, zorder=5,
                bbox=dict(boxstyle="round,pad=0.4", fc=P.SURFACE, ec=P.GRID, lw=0.8, alpha=0.92))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, d["value_at_risk"].max() * 1.15)
    ax.set_xlabel("Predicted churn probability (out-of-sample)")
    ax.set_ylabel("Margin at risk per customer ($)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(loc="lower center", ncols=2, markerscale=2.5, framealpha=0.92, facecolor=P.SURFACE, frameon=True,
              edgecolor="none")
    P.title(ax, "Risk x value matrix - every customer, scored out-of-sample",
            f"Vertical line = profit-optimal risk threshold ({risk_cut:.2f}); horizontal = median margin at risk")
    P.savefig(fig, "segmentation_risk_value_matrix")


def _persona_name(r: pd.Series) -> str:
    tenure = "New" if r["tenure"] < 15 else ("Established" if r["tenure"] < 45 else "Loyal")
    if r["has_internet"] < 0.5:
        net = "phone-only"
    elif r["is_fiber"] > 0.5:
        net = "fibre"
    else:
        net = "DSL"
    contract = "month-to-month" if r["contract_months"] < 6 else ("1-yr" if r["contract_months"] < 18 else "2-yr")
    extras = "high-engagement" if r["num_services"] >= 5 else ("low-engagement" if r["num_services"] <= 2 else "")
    return " ".join(x for x in [tenure, net, contract, extras] if x).replace("  ", " ")


def personas(df: pd.DataFrame, k_range=range(4, 9), seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    fe = FeatureEngineer().transform(df)
    cols = ["tenure", "MonthlyCharges", "num_services", "num_protection", "num_streaming",
            "contract_months", "auto_payment", "is_fiber", "has_internet", "has_family", "SeniorCitizen"]
    Z = StandardScaler().fit_transform(fe[cols])
    sil = {}
    for k in k_range:
        labels = KMeans(k, n_init=10, random_state=seed).fit_predict(Z)
        sil[k] = float(silhouette_score(Z, labels, sample_size=3000, random_state=seed))
    best_k = max(sil, key=sil.get)
    km = KMeans(best_k, n_init=20, random_state=seed).fit(Z)
    fe["persona_id"] = km.labels_
    fe[TARGET] = df[TARGET].values
    prof = fe.groupby("persona_id")[cols + [TARGET]].mean()
    prof["customers"] = fe.groupby("persona_id").size()
    prof["persona"] = prof.apply(_persona_name, axis=1)
    # de-duplicate names
    dup = prof["persona"].duplicated(keep=False)
    prof.loc[dup, "persona"] = prof.loc[dup].apply(lambda r: f"{r['persona']} (${r['MonthlyCharges']:.0f}/mo)", axis=1)
    prof = prof.sort_values(TARGET, ascending=False)
    assign = pd.DataFrame({"customerID": df["customerID"].values,
                           "persona": prof["persona"].reindex(km.labels_).values})
    return prof, assign, sil


def fig_personas(prof: pd.DataFrame, overall: float) -> None:
    d = prof.iloc[::-1]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17, 0.55 * len(d) + 1.8), sharey=True,
                                     width_ratios=[1.3, 1, 1], layout="constrained")
    a1.barh(d["persona"], d[TARGET], color=P.CHURNED, height=0.62)
    a1.axvline(overall, color=P.INK_2, lw=1)
    for i, v in enumerate(d[TARGET]):
        a1.text(v + 0.01, i, P.pct(v, 0), va="center", fontsize=8.5, color=P.INK_2)
    a1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    a1.set_title("Churn rate", fontsize=10.5)
    a2.barh(d["persona"], d["customers"], color=P.BLUE, height=0.62)
    for i, v in enumerate(d["customers"]):
        a2.text(v + 20, i, f"{int(v):,}", va="center", fontsize=8.5, color=P.INK_2)
    a2.set_title("Customers", fontsize=10.5)
    a3.barh(d["persona"], d["MonthlyCharges"], color=P.BLUE, height=0.62)
    for i, v in enumerate(d["MonthlyCharges"]):
        a3.text(v + 1, i, f"${v:.0f}", va="center", fontsize=8.5, color=P.INK_2)
    a3.set_title("Avg monthly charges", fontsize=10.5)
    for a in (a1, a2, a3):
        a.grid(axis="y", visible=False)
    fig.suptitle(f"Behavioural personas (K-Means, k={len(d)} chosen by silhouette over k=4-8)", x=0.01, ha="left",
                 fontsize=13, fontweight="semibold")
    P.savefig(fig, "segmentation_personas")


def run(df: pd.DataFrame, p: np.ndarray, value: np.ndarray, risk_cut: float) -> dict:
    d = risk_value_matrix(df, p, value, risk_cut)
    summary = segment_summary(d)
    fig_risk_value(d, summary, risk_cut)
    prof, assign, sil = personas(df)
    fig_personas(prof, df[TARGET].mean())
    d = d.merge(assign, on="customerID")
    d.to_csv(TABLES_DIR / "customers_scored.csv", index=False)
    summary.to_csv(TABLES_DIR / "segment_summary.csv")
    prof.to_csv(TABLES_DIR / "persona_profiles.csv")
    return {
        "segments": summary.drop(columns="playbook").round(4).reset_index().to_dict("records"),
        "n_personas": int(len(prof)),
        "silhouette": sil,
        "personas": prof[["persona", "customers", TARGET, "MonthlyCharges", "tenure"]].round(3).to_dict("records"),
    }
