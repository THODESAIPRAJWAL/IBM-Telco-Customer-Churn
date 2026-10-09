"""Exploratory & statistical analysis.

Goes beyond "plot everything": every relationship is backed by a hypothesis test
and an effect size, so the report can say *how much* a factor matters, not just
that it looks different.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.feature_selection import mutual_info_classif

from . import plotting as P
from .config import CATEGORY_LEVELS, TABLES_DIR, TARGET
from .features import CATEGORICAL_FEATURES, FeatureEngineer

plt = P.plt

CAT_COLS = ["SeniorCitizen", *CATEGORY_LEVELS.keys()]
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CONTRACT_ORDER = ["Month-to-month", "One year", "Two year"]
# Contract length is ordinal -> ordinal blue ramp (validated steps 250 / 450 / 650)
CONTRACT_COLORS = dict(zip(CONTRACT_ORDER, ["#86b6ef", "#2a78d6", "#104281"]))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramer's V (Bergsma, 2013)."""
    table = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.values.sum()
    r, k = table.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    r_c, k_c = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denom = min(k_c - 1, r_c - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """Eta: how much of a numeric variable's variance a categorical explains."""
    df = pd.DataFrame({"c": categories.astype(str), "v": values.astype(float)})
    grand = df["v"].mean()
    between = df.groupby("c")["v"].agg(["count", "mean"])
    ss_between = (between["count"] * (between["mean"] - grand) ** 2).sum()
    ss_total = ((df["v"] - grand) ** 2).sum()
    return float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else 0.0


def categorical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Chi-square test of independence + Cramer's V for every categorical."""
    rows = []
    for col in CAT_COLS:
        table = pd.crosstab(df[col], df[TARGET])
        chi2, p, dof, _ = stats.chi2_contingency(table)
        rates = df.groupby(col)[TARGET].mean()
        rows.append(
            {
                "feature": col,
                "chi2": chi2,
                "dof": dof,
                "p_value": p,
                "cramers_v": cramers_v(df[col], df[TARGET]),
                "highest_churn_level": rates.idxmax(),
                "highest_churn_rate": rates.max(),
                "lowest_churn_level": rates.idxmin(),
                "lowest_churn_rate": rates.min(),
            }
        )
    out = pd.DataFrame(rows).sort_values("cramers_v", ascending=False)
    # Bonferroni correction across the family of tests
    out["p_bonferroni"] = np.minimum(out["p_value"] * len(out), 1.0)
    out["significant"] = out["p_bonferroni"] < 0.05
    return out.reset_index(drop=True)


def numeric_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Mann-Whitney U (distribution-free) with rank-biserial effect size."""
    rows = []
    for col in NUM_COLS:
        churned, stayed = df.loc[df[TARGET] == 1, col], df.loc[df[TARGET] == 0, col]
        u, p = stats.mannwhitneyu(churned, stayed, alternative="two-sided")
        rank_biserial = 1 - 2 * u / (len(churned) * len(stayed))
        rows.append(
            {
                "feature": col,
                "median_churned": churned.median(),
                "median_retained": stayed.median(),
                "mean_churned": churned.mean(),
                "mean_retained": stayed.mean(),
                "mann_whitney_u": u,
                "p_value": p,
                "rank_biserial": -rank_biserial,  # >0 => larger values in churned group
                "point_biserial_r": stats.pointbiserialr(df[TARGET], df[col])[0],
            }
        )
    return pd.DataFrame(rows)


def mutual_information(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Mutual information of raw + engineered features with churn."""
    fe = FeatureEngineer(drop=()).transform(df)
    fe["gender"] = df["gender"].values
    X = fe.copy()
    discrete = []
    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype) in ("str", "string", "category"):
            X[col] = X[col].astype("category").cat.codes
            discrete.append(True)
        else:
            discrete.append(X[col].nunique() <= 10)
    mi = mutual_info_classif(X, df[TARGET], discrete_features=discrete, random_state=random_state)
    return (
        pd.DataFrame({"feature": X.columns, "mutual_information": mi})
        .sort_values("mutual_information", ascending=False)
        .reset_index(drop=True)
    )


def association_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Mixed-type association matrix in [0, 1]: Cramer's V (cat-cat),
    correlation ratio (cat-num) and |Spearman| (num-num)."""
    cols = [TARGET, *NUM_COLS, *CAT_COLS]
    data = df[cols].copy()
    data[TARGET] = data[TARGET].map({1: "Yes", 0: "No"})
    is_num = {c: c in NUM_COLS for c in cols}
    m = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            if is_num[a] and is_num[b]:
                v = abs(stats.spearmanr(data[a], data[b])[0])
            elif is_num[a]:
                v = correlation_ratio(data[b], data[a])
            elif is_num[b]:
                v = correlation_ratio(data[a], data[b])
            else:
                v = cramers_v(data[a], data[b])
            m.loc[a, b] = m.loc[b, a] = v
    return m


def inferential_logit(df: pd.DataFrame):
    """Classical logistic regression for *inference* (odds ratios, CIs, p-values).

    Uses a curated, non-collinear design: "No internet service" levels are folded
    into "No" (the information lives in InternetService), and MonthlyCharges /
    TotalCharges are omitted because they are near-deterministic functions of the
    services held and tenure.
    """
    d = pd.DataFrame(index=df.index)
    d["Contract: One year"] = (df["Contract"] == "One year").astype(int)
    d["Contract: Two year"] = (df["Contract"] == "Two year").astype(int)
    d["Internet: Fiber optic"] = (df["InternetService"] == "Fiber optic").astype(int)
    d["Internet: None"] = (df["InternetService"] == "No").astype(int)
    d["Pay: Electronic check"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    d["Pay: Mailed check"] = (df["PaymentMethod"] == "Mailed check").astype(int)
    d["Paperless billing"] = (df["PaperlessBilling"] == "Yes").astype(int)
    d["Senior citizen"] = df["SeniorCitizen"]
    d["Partner"] = (df["Partner"] == "Yes").astype(int)
    d["Dependents"] = (df["Dependents"] == "Yes").astype(int)
    for svc in [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "MultipleLines",
    ]:
        d[svc] = (df[svc] == "Yes").astype(int)
    d["Tenure (per doubling)"] = np.log2(df["tenure"] + 1)
    X = sm.add_constant(d.astype(float))
    model = sm.Logit(df[TARGET].astype(float), X).fit(disp=False)
    ci = model.conf_int()
    table = pd.DataFrame(
        {
            "odds_ratio": np.exp(model.params),
            "ci_low": np.exp(ci[0]),
            "ci_high": np.exp(ci[1]),
            "p_value": model.pvalues,
        }
    ).drop(index="const")
    table.index.name = "term"
    return model, table.sort_values("odds_ratio", ascending=False)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_churn_rate_by_category(df: pd.DataFrame, cat_tests: pd.DataFrame) -> None:
    overall = df[TARGET].mean()
    features = cat_tests["feature"].tolist()
    ncols = 4
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 2.35 * nrows), layout="constrained")
    for ax, feat in zip(axes.flat, features):
        rates = df.groupby(feat)[TARGET].agg(["mean", "size"]).sort_values("mean")
        labels = [str(i).replace(" (automatic)", " (auto)") for i in rates.index]
        if feat == "SeniorCitizen":
            labels = ["No" if l == "0" else "Yes" for l in labels]
        ax.barh(labels, rates["mean"], color=P.CHURNED, height=0.62)
        ax.axvline(overall, color=P.INK_2, lw=1)
        for y, (r, n) in enumerate(zip(rates["mean"], rates["size"])):
            ax.text(r + 0.01, y, f"{P.pct(r, 0)}  (n={n:,})", va="center", fontsize=8, color=P.INK_2)
        v = cat_tests.set_index("feature").loc[feat, "cramers_v"]
        ax.set_title(f"{feat}   V={v:.2f}", fontsize=10, pad=6)
        ax.set_xlim(0, 0.75)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=8.5)
    for ax in list(axes.flat)[len(features) :]:
        ax.set_visible(False)
    fig.suptitle(
        "Churn rate by customer attribute (sorted by strength of association, Cramer's V)",
        x=0.01, ha="left", fontsize=13, fontweight="semibold",
    )
    fig.text(0.01, -0.01, f"Vertical line = overall churn rate ({P.pct(overall)}).",
             fontsize=9, color=P.INK_2)
    P.savefig(fig, "eda_churn_rate_by_category")


def fig_numeric_distributions(df: pd.DataFrame, num_tests: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    t = num_tests.set_index("feature")
    for ax, col in zip(axes, NUM_COLS):
        bins = np.histogram_bin_edges(df[col], bins=36)
        for label, name in [(0, "Retained"), (1, "Churned")]:
            ax.hist(df.loc[df[TARGET] == label, col], bins=bins, density=True, alpha=0.55,
                    color=P.CHURN_COLORS[label], label=name, edgecolor=P.SURFACE, linewidth=1)
        for label in (0, 1):
            med = df.loc[df[TARGET] == label, col].median()
            ax.axvline(med, color=P.CHURN_COLORS[label], lw=1.5)
        rb = t.loc[col, "rank_biserial"]
        P.title(ax, col, f"medians {t.loc[col, 'median_retained']:.0f} vs {t.loc[col, 'median_churned']:.0f}"
                f"  |  rank-biserial r = {rb:+.2f}")
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    axes[0].legend(loc="upper right")
    P.savefig(fig, "eda_numeric_distributions")


def fig_tenure_contract(df: pd.DataFrame) -> None:
    d = df.copy()
    d["tenure_bin"] = pd.cut(d["tenure"], bins=range(-1, 79, 6))
    fig, ax = plt.subplots(figsize=(10, 4.6), layout="constrained")
    for contract in CONTRACT_ORDER:
        g = d[d["Contract"] == contract].groupby("tenure_bin", observed=True)[TARGET].agg(["mean", "size"])
        g = g[g["size"] >= 20]
        x = [iv.right for iv in g.index]
        ax.plot(x, g["mean"], color=CONTRACT_COLORS[contract], marker="o", ms=5, label=contract)
        ax.text(x[-1] + 1, g["mean"].iloc[-1], contract, color=P.INK_2, va="center", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlabel("Tenure (months, 6-month bins)")
    ax.set_ylabel("Churn rate")
    ax.set_xlim(0, 84)
    ax.legend(loc="upper right")
    new = df[df["tenure"] <= 6].groupby("Contract")[TARGET].mean()
    P.title(ax, "Churn falls with tenure, but contract type dominates",
            f"In their first 6 months, {P.pct(new['Month-to-month'], 0)} of month-to-month customers churn "
            f"vs {P.pct(new['One year'], 0)} on one-year and {P.pct(new['Two year'], 0)} on two-year contracts")
    P.savefig(fig, "eda_churn_by_tenure_and_contract")


def fig_association_heatmap(assoc: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 9.5), layout="constrained")
    im = ax.imshow(assoc.values, cmap=P.SEQUENTIAL, vmin=0, vmax=1)
    ax.set_xticks(range(len(assoc)), assoc.columns, rotation=60, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(assoc)), assoc.index, fontsize=8.5)
    ax.grid(False)
    for i in range(len(assoc)):
        for j in range(len(assoc)):
            v = assoc.values[i, j]
            if i != j and v >= 0.3:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > 0.6 else P.INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.outline.set_visible(False)
    cb.set_label("Association strength")
    P.title(ax, "Mixed-type association matrix",
            "Cramer's V (categorical pairs), correlation ratio (categorical-numeric), |Spearman| (numeric pairs)")
    P.savefig(fig, "eda_association_matrix")


def fig_mutual_information(mi: pd.DataFrame, top: int = 20) -> None:
    d = mi.head(top).iloc[::-1]
    engineered = set(d["feature"]) - set(CAT_COLS) - set(NUM_COLS)
    fig, ax = plt.subplots(figsize=(9, 6.5), layout="constrained")
    colors = [P.BLUE if f in engineered else P.GHOST for f in d["feature"]]
    ax.barh(d["feature"], d["mutual_information"], color=colors, height=0.65)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mutual information with churn (nats)")
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=P.BLUE, label="Engineered feature"),
                       Patch(color=P.GHOST, label="Original feature")], loc="lower right")
    P.title(ax, f"Top {top} features by mutual information",
            "Engineered features (blue) rank alongside the strongest raw signals")
    P.savefig(fig, "eda_mutual_information")


def fig_odds_ratios(table: pd.DataFrame) -> None:
    d = table.sort_values("odds_ratio")
    fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
    y = np.arange(len(d))
    raises = d["odds_ratio"] > 1
    sig = d["p_value"] < 0.05
    for i, (_, r) in enumerate(d.iterrows()):
        c = (P.CHURNED if r["odds_ratio"] > 1 else P.RETAINED) if r["p_value"] < 0.05 else P.GHOST
        ax.plot([r["ci_low"], r["ci_high"]], [i, i], color=c, lw=2, solid_capstyle="round")
        ax.plot(r["odds_ratio"], i, "o", color=c, ms=8, markeredgecolor=P.SURFACE, markeredgewidth=2)
    ax.axvline(1, color=P.INK_2, lw=1)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 4], ["0.1", "0.2", "0.5", "1", "2", "4"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yticks(y, [f"{t}{'' if s else '  (n.s.)'}" for t, s in zip(d.index, sig)])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Odds ratio (log scale) with 95% CI")
    from matplotlib.lines import Line2D

    ax.legend(handles=[Line2D([], [], color=P.CHURNED, marker="o", label="Raises churn odds"),
                       Line2D([], [], color=P.RETAINED, marker="o", label="Lowers churn odds"),
                       Line2D([], [], color=P.GHOST, marker="o", label="Not significant")],
              loc="lower right")
    n_up, n_down = int((raises & sig).sum()), int((~raises & sig).sum())
    P.title(ax, "What drives churn? Adjusted odds ratios (logistic regression)",
            f"{n_up} factors significantly raise and {n_down} lower churn odds, holding all others constant")
    P.savefig(fig, "eda_odds_ratios")


def fig_tenure_vs_charges(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    for label, name in [(0, "Retained"), (1, "Churned")]:
        d = df[df[TARGET] == label]
        ax.scatter(d["tenure"], d["MonthlyCharges"], s=9, alpha=0.45 if label else 0.3,
                   color=P.CHURN_COLORS[label], label=name, linewidths=0)
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Monthly charges ($)")
    ax.legend(loc="lower right", markerscale=2.5)
    P.title(ax, "Churners cluster at short tenure and high monthly charges",
            "The top-left corner (new customers paying $70+) is the highest-risk region")
    P.savefig(fig, "eda_tenure_vs_charges")


def fig_revenue_at_risk(df: pd.DataFrame) -> pd.DataFrame:
    churned = df[df[TARGET] == 1]
    by = (churned.groupby("Contract")["MonthlyCharges"].sum()
          .reindex(CONTRACT_ORDER).rename("monthly_revenue_lost").to_frame())
    by["share"] = by["monthly_revenue_lost"] / by["monthly_revenue_lost"].sum()
    fig, ax = plt.subplots(figsize=(9, 3.4), layout="constrained")
    ax.barh(by.index[::-1], by["monthly_revenue_lost"][::-1],
            color=[CONTRACT_COLORS[c] for c in by.index[::-1]], height=0.6)
    for i, (c, r) in enumerate(by.iloc[::-1].iterrows()):
        ax.text(r["monthly_revenue_lost"] + 1500, i,
                f"${r['monthly_revenue_lost']:,.0f}/mo  ({P.pct(r['share'], 0)})",
                va="center", fontsize=9, color=P.INK_2)
    ax.set_xlim(0, by["monthly_revenue_lost"].max() * 1.35)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    ax.grid(axis="y", visible=False)
    total = by["monthly_revenue_lost"].sum()
    P.title(ax, f"${total:,.0f} of monthly recurring revenue walked out the door",
            f"{P.pct(by.loc['Month-to-month', 'share'], 0)} of lost revenue came from month-to-month contracts")
    P.savefig(fig, "eda_revenue_at_risk")
    return by


def fig_services_engagement(df: pd.DataFrame) -> None:
    fe = FeatureEngineer(drop=()).transform(df)
    d = pd.DataFrame({"n": fe["num_protection"], "internet": df["InternetService"], "y": df[TARGET]})
    d = d[d["internet"] != "No"]
    g = d.groupby("n")["y"].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
    ax.bar(g.index, g["mean"], color=P.CHURNED, width=0.6)
    for x, (r, n) in zip(g.index, g.values):
        ax.text(x, r + 0.01, f"{P.pct(r, 0)}", ha="center", fontsize=9, color=P.INK_2)
    ax.set_xticks(g.index, [f"{i}\n(n={int(n):,})" for i, n in zip(g.index, g["size"])])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlabel("Number of protection services (security, backup, device protection, tech support)")
    ax.grid(axis="x", visible=False)
    P.title(ax, "Every protection add-on lowers churn",
            "Internet customers only - a strong, monotonic engagement effect")
    P.savefig(fig, "eda_protection_services")


# ---------------------------------------------------------------------------
def run(df: pd.DataFrame) -> dict:
    cat = categorical_tests(df)
    num = numeric_tests(df)
    mi = mutual_information(df)
    assoc = association_matrix(df)
    _, odds = inferential_logit(df)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    cat.to_csv(TABLES_DIR / "stats_categorical_tests.csv", index=False)
    num.to_csv(TABLES_DIR / "stats_numeric_tests.csv", index=False)
    mi.to_csv(TABLES_DIR / "stats_mutual_information.csv", index=False)
    odds.to_csv(TABLES_DIR / "stats_logit_odds_ratios.csv")

    fig_churn_rate_by_category(df, cat)
    fig_numeric_distributions(df, num)
    fig_tenure_contract(df)
    fig_association_heatmap(assoc)
    fig_mutual_information(mi)
    fig_odds_ratios(odds)
    fig_tenure_vs_charges(df)
    revenue = fig_revenue_at_risk(df)
    fig_services_engagement(df)

    by_contract = df.groupby("Contract")[TARGET].mean()
    return {
        "n_customers": int(len(df)),
        "churn_rate": float(df[TARGET].mean()),
        "monthly_revenue_lost": float(df.loc[df[TARGET] == 1, "MonthlyCharges"].sum()),
        "monthly_revenue_total": float(df["MonthlyCharges"].sum()),
        "churn_by_contract": by_contract.to_dict(),
        "mtm_vs_one_year_ratio": float(by_contract["Month-to-month"] / by_contract["One year"]),
        "mtm_vs_two_year_ratio": float(by_contract["Month-to-month"] / by_contract["Two year"]),
        "top_categorical": cat.head(5)[["feature", "cramers_v"]].to_dict("records"),
        "non_significant_features": cat.loc[~cat["significant"], "feature"].tolist(),
        "revenue_lost_by_contract": revenue["monthly_revenue_lost"].to_dict(),
        "gender_p_value": float(cat.set_index("feature").loc["gender", "p_value"]),
    }


__all__ = ["run", "CATEGORICAL_FEATURES"]
