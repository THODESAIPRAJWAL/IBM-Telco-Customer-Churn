"""Survival analysis: churn as a time-to-event problem.

A classifier answers *whether* a customer churns; survival analysis answers
*when*. ``tenure`` is the observed lifetime, ``Churn`` the event indicator and
active customers are right-censored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test

from . import plotting as P
from .config import TABLES_DIR, TARGET

plt = P.plt

CONTRACT_ORDER = ["Month-to-month", "One year", "Two year"]
CONTRACT_COLORS = dict(zip(CONTRACT_ORDER, ["#86b6ef", "#2a78d6", "#104281"]))  # ordinal ramp


def _duration(df: pd.DataFrame) -> pd.Series:
    # tenure==0 customers joined this month; give them a tiny positive duration
    return df["tenure"].clip(lower=0.5)


def km_by(df: pd.DataFrame, col: str) -> tuple[dict[str, KaplanMeierFitter], float]:
    fitters = {}
    for level, g in df.groupby(col):
        kmf = KaplanMeierFitter(label=str(level))
        kmf.fit(_duration(g), g[TARGET])
        fitters[str(level)] = kmf
    lr = multivariate_logrank_test(_duration(df), df[col].astype(str), df[TARGET])
    return fitters, float(lr.p_value)


def cox_design(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame(index=df.index)
    d["duration"] = _duration(df)
    d["event"] = df[TARGET]
    d["Contract: One year"] = (df["Contract"] == "One year").astype(int)
    d["Contract: Two year"] = (df["Contract"] == "Two year").astype(int)
    d["Internet: Fiber optic"] = (df["InternetService"] == "Fiber optic").astype(int)
    d["Internet: None"] = (df["InternetService"] == "No").astype(int)
    d["Pay: Electronic check"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    d["Pay: Automatic"] = df["PaymentMethod"].str.contains("automatic").astype(int)
    d["Paperless billing"] = (df["PaperlessBilling"] == "Yes").astype(int)
    d["Senior citizen"] = df["SeniorCitizen"]
    d["Partner"] = (df["Partner"] == "Yes").astype(int)
    d["Dependents"] = (df["Dependents"] == "Yes").astype(int)
    for svc in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
        d[svc] = (df[svc] == "Yes").astype(int)
    d["Monthly charges (per $10)"] = df["MonthlyCharges"] / 10
    return d


def fit_cox(df: pd.DataFrame) -> tuple[CoxPHFitter, pd.DataFrame]:
    d = cox_design(df)
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(d, duration_col="duration", event_col="event")
    ph = proportional_hazard_test(cph, d, time_transform="rank")
    ph_table = ph.summary[["test_statistic", "p"]].rename(columns={"p": "ph_test_p"})
    return cph, ph_table


# ---------------------------------------------------------------------------
def fig_km(df: pd.DataFrame) -> dict[str, float]:
    specs = [
        ("Contract", CONTRACT_ORDER, CONTRACT_COLORS),
        ("InternetService", ["DSL", "Fiber optic", "No"], dict(zip(["DSL", "Fiber optic", "No"], P.CATEGORICAL[:3]))),
        ("PaymentMethod", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
         dict(zip(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], P.CATEGORICAL[:4]))),
    ]
    pvals = {}
    floor = min(f.survival_function_.iloc[:, 0].min() for col, *_ in specs for f in km_by(df, col)[0].values())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True, layout="constrained")
    for ax, (col, order, colors) in zip(axes, specs):
        fitters, p = km_by(df, col)
        pvals[col] = p
        for level in order:
            kmf = fitters[level]
            sf = kmf.survival_function_
            ci = kmf.confidence_interval_
            ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color=colors[level], alpha=0.12, step="post", lw=0)
            ax.step(sf.index, sf.iloc[:, 0], where="post", color=colors[level],
                    label=level.replace(" (automatic)", " (auto)"))
        ax.set_xlabel("Tenure (months)")
        ax.set_ylim(np.floor(floor * 10) / 10, 1.01)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.legend(loc="lower left", fontsize=8.5)
        P.title(ax, f"By {col}", f"log-rank p = {p:.1e}")
    axes[0].set_ylabel("Share of customers still active")
    fig.suptitle("Kaplan-Meier retention curves (shaded = 95% CI)", x=0.01, ha="left", fontsize=13, fontweight="semibold")
    P.savefig(fig, "survival_kaplan_meier")
    return pvals


def fig_hazard_ratios(cph: CoxPHFitter, ph_table: pd.DataFrame) -> pd.DataFrame:
    s = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].copy()
    s.columns = ["hazard_ratio", "ci_low", "ci_high", "p_value"]
    s = s.join(ph_table["ph_test_p"])
    s = s.sort_values("hazard_ratio")
    fig, ax = plt.subplots(figsize=(9, 6.5), layout="constrained")
    for i, (name, r) in enumerate(s.iterrows()):
        c = (P.CHURNED if r["hazard_ratio"] > 1 else P.RETAINED) if r["p_value"] < 0.05 else P.GHOST
        ax.plot([r["ci_low"], r["ci_high"]], [i, i], color=c, lw=2, solid_capstyle="round")
        ax.plot(r["hazard_ratio"], i, "o", color=c, ms=8, markeredgecolor=P.SURFACE, markeredgewidth=2)
    ax.axvline(1, color=P.INK_2, lw=1)
    ax.set_xscale("log")
    ticks = [0.05, 0.1, 0.2, 0.5, 1, 2]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yticks(range(len(s)), s.index)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Hazard ratio (log scale) with 95% CI")
    P.title(ax, "Cox proportional hazards: what accelerates churn?",
            f"Concordance index = {cph.concordance_index_:.3f}; HR < 1 means customers stay longer")
    P.savefig(fig, "survival_cox_hazard_ratios")
    return s


def fig_expected_lifetime(cph: CoxPHFitter, df: pd.DataFrame) -> pd.DataFrame:
    """Survival curves for three archetypal customers built from the Cox model."""
    design = cox_design(df).drop(columns=["duration", "event"])
    base = design.median().to_frame().T
    base[:] = 0
    base["Monthly charges (per $10)"] = 7.0
    base["Internet: Fiber optic"] = 1
    base["Paperless billing"] = 1
    archetypes = {
        "Month-to-month, e-check, fibre, no add-ons": dict(**{"Pay: Electronic check": 1}),
        "One-year contract, auto-pay, security + support": {"Contract: One year": 1, "Pay: Automatic": 1,
                                                          "OnlineSecurity": 1, "TechSupport": 1},
        "Two-year contract, auto-pay, security + support": {"Contract: Two year": 1, "Pay: Automatic": 1,
                                                          "OnlineSecurity": 1, "TechSupport": 1},
    }
    rows = []
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    t = np.arange(0, 73)
    for (name, changes), color in zip(archetypes.items(), CONTRACT_COLORS.values()):
        x = base.copy()
        for k, v in changes.items():
            x[k] = v
        sf = cph.predict_survival_function(x, times=t).iloc[:, 0]
        ax.plot(t, sf.values, color=color, label=name)
        rmst = float(np.trapezoid(sf.values, t))
        rows.append({"archetype": name, "survival_12m": float(sf.loc[12]), "survival_24m": float(sf.loc[24]),
                     "expected_months_in_first_72": rmst})
    ax.set_xlabel("Months since joining"); ax.set_ylabel("Probability still a customer")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    r = pd.DataFrame(rows)
    P.title(ax, "Model-implied retention for three customer archetypes ($70/month)",
            f"12-month retention: {P.pct(r['survival_12m'].iloc[0], 0)} vs {P.pct(r['survival_12m'].iloc[1], 0)} "
            f"vs {P.pct(r['survival_12m'].iloc[2], 0)}")
    P.savefig(fig, "survival_archetypes")
    return r


def run(df: pd.DataFrame) -> dict:
    logrank = fig_km(df)
    cph, ph = fit_cox(df)
    hr = fig_hazard_ratios(cph, ph)
    arche = fig_expected_lifetime(cph, df)
    hr.to_csv(TABLES_DIR / "survival_cox_hazard_ratios.csv")
    arche.to_csv(TABLES_DIR / "survival_archetypes.csv", index=False)
    medians = {}
    fitters, _ = km_by(df, "Contract")
    for k, f in fitters.items():
        m = f.median_survival_time_
        medians[k] = None if not np.isfinite(m) else float(m)
    km_all = KaplanMeierFitter().fit(_duration(df), df[TARGET])
    return {
        "logrank_p": logrank,
        "cox_concordance": float(cph.concordance_index_),
        "median_lifetime_by_contract": medians,
        "retention_12m_overall": float(km_all.predict(12)),
        "retention_24m_overall": float(km_all.predict(24)),
        "ph_violations": hr.index[hr["ph_test_p"] < 0.01].tolist(),
        "archetypes": arche.to_dict("records"),
    }
