import altair as alt
import pandas as pd
import streamlit as st

import lib

lib.require_artifacts()
m = lib.metrics()
md, e, sv = m["metadata"], m["eda"], m["survival"]
tm = md["test_metrics"]
scale = m["business"]["scale_to_full_base"]
pol = pd.DataFrame(m["business"]["policies_test"]).set_index("policy")
ev = pol.loc["Value-aware rule (p*s*v > c)"]
everyone = pol.loc["Target everyone"]
df = lib.customers()

st.caption(
    f"IBM Telco Customer Churn - {e['n_customers']:,} customers. Deployed model: **{md['model_name']}**, "
    f"trained {md['trained_at'][:10]}."
)

with st.container(horizontal=True):
    st.metric("Churn rate", f"{e['churn_rate']:.1%}", border=True,
              help="Share of customers who left in the observation month")
    st.metric("Monthly revenue lost", lib.money(e["monthly_revenue_lost"]),
              f"{e['monthly_revenue_lost'] / e['monthly_revenue_total']:.0%} of MRR", delta_color="off", delta_arrow="off", border=True)
    st.metric("Model ROC-AUC (hold-out)", f"{tm['roc_auc']:.3f}",
              f"95% CI {tm['auc_ci_low']:.3f}-{tm['auc_ci_high']:.3f}", delta_color="off", delta_arrow="off", border=True)
    st.metric("Top-decile lift", f"{tm['lift_top10']:.1f}x", "vs. random targeting", delta_color="off", delta_arrow="off", border=True)
    st.metric("Projected campaign profit", lib.money(ev["net_profit"] * scale),
              f"vs {lib.money(everyone['net_profit'] * scale)} contacting everyone", delta_color="off", delta_arrow="off", border=True,
              help="Value-aware targeting, hold-out results scaled to the full customer base")

left, right = st.columns([1.15, 1])
with left:
    with st.container(border=True):
        st.subheader("Churn falls with tenure - but contract type dominates")
        d = df.assign(tenure_bin=(df["tenure"] // 6) * 6 + 3)
        g = d.groupby(["Contract", "tenure_bin"])["Churn"].agg(churn_rate="mean", customers="size").reset_index()
        g = g[g["customers"] >= 20]
        order = ["Month-to-month", "One year", "Two year"]
        chart = alt.Chart(g).mark_line(point=alt.OverlayMarkDef(size=45), strokeWidth=2).encode(
            x=alt.X("tenure_bin:Q", title="Tenure (months)"),
            y=alt.Y("churn_rate:Q", title="Churn rate", axis=alt.Axis(format="%")),
            color=alt.Color("Contract:N", sort=order,
                            scale=alt.Scale(domain=order, range=["#86b6ef", "#2a78d6", "#104281"]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["Contract", alt.Tooltip("tenure_bin:Q", title="Tenure (mid-bin)"),
                     alt.Tooltip("churn_rate:Q", format=".1%", title="Churn rate"),
                     alt.Tooltip("customers:Q", format=",")],
        ).properties(height=320)
        st.altair_chart(chart)
        st.caption(f"Month-to-month customers churn {e['mtm_vs_one_year_ratio']:.1f}x more than one-year and "
                   f"{e['mtm_vs_two_year_ratio']:.0f}x more than two-year customers.")
with right:
    with st.container(border=True):
        st.subheader("Where the lost revenue comes from")
        rev = pd.DataFrame({"Contract": list(e["revenue_lost_by_contract"].keys()),
                            "lost": list(e["revenue_lost_by_contract"].values())})
        chart = alt.Chart(rev).mark_bar(cornerRadiusEnd=4, height={"band": 0.6}).encode(
            y=alt.Y("Contract:N", sort=["Month-to-month", "One year", "Two year"], title=None),
            x=alt.X("lost:Q", title="Monthly revenue lost ($)", axis=alt.Axis(format="$,.0f")),
            color=alt.Color("Contract:N", scale=alt.Scale(domain=["Month-to-month", "One year", "Two year"],
                                                          range=["#86b6ef", "#2a78d6", "#104281"]), legend=None),
            tooltip=["Contract", alt.Tooltip("lost:Q", format="$,.0f", title="Monthly revenue lost")],
        ).properties(height=180)
        st.altair_chart(chart)
        st.caption(f"Only {sv['retention_12m_overall']:.0%} of customers are still active after 12 months "
                   f"(Kaplan-Meier); the median month-to-month customer leaves after "
                   f"{sv['median_lifetime_by_contract']['Month-to-month']:.0f} months.")

st.subheader("What the analysis found", icon=":material/lightbulb:")
top = ", ".join(f"{d['feature']}" for d in m["explainability"]["shap_top_features"][:4])
hr_two_year = lib.table("survival_cox_hazard_ratios").set_index("covariate").loc["Contract: Two year", "hazard_ratio"]
with st.container(border=True):
    st.markdown(f"""
- **Commitment is the #1 lever.** Contract type has the strongest association with churn (Cramer's V
  {e['top_categorical'][0]['cramers_v']:.2f}); in the Cox survival model a two-year contract cuts the churn hazard
  by {1 - hr_two_year:.0%} relative to month-to-month, holding everything else constant.
- **Service gaps drive risk.** Fibre customers without online security or tech support, and customers paying by
  electronic check, are the highest-risk groups even after adjusting for contract and tenure.
- **The model's top drivers** (SHAP, raw features): {top}.
- **Targeting beats blanket offers.** Contacting only customers whose expected saved margin exceeds the offer cost
  yields **{lib.money(ev['net_profit'] * scale)}** projected profit vs **{lib.money(everyone['net_profit'] * scale)}**
  for contacting everyone.
- **Fair by design.** Gender has no relationship with churn (p = {e['gender_p_value']:.2f}) and is excluded from the model.
""")
