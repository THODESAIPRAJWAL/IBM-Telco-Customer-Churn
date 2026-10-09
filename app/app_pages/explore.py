import altair as alt
import streamlit as st

import lib

lib.require_artifacts()
df = lib.customers()

st.caption("Slice the customer base and see how churn changes. Every chart below respects the filters.")

with st.container(border=True, horizontal=True):
    contracts = st.multiselect("Contract", lib.CATEGORY_LEVELS["Contract"], default=lib.CATEGORY_LEVELS["Contract"])
    internet = st.multiselect("Internet service", lib.CATEGORY_LEVELS["InternetService"],
                              default=lib.CATEGORY_LEVELS["InternetService"])
    tenure = st.slider("Tenure (months)", 0, 72, (0, 72))
    senior = st.segmented_control("Senior citizen", ["All", "Yes", "No"], default="All")

f = df[df["Contract"].isin(contracts) & df["InternetService"].isin(internet) & df["tenure"].between(*tenure)]
if senior != "All":
    f = f[f["SeniorCitizen"] == (1 if senior == "Yes" else 0)]
if f.empty:
    st.warning("No customers match these filters.", icon=":material/filter_alt_off:")
    st.stop()

overall = f["Churn"].mean()
with st.container(horizontal=True):
    st.metric("Customers in slice", f"{len(f):,}", border=True)
    st.metric("Churn rate in slice", f"{overall:.1%}", f"{overall - df['Churn'].mean():+.1%} vs all customers",
              delta_color="inverse", border=True)
    st.metric("Avg monthly charges", f"${f['MonthlyCharges'].mean():.0f}", border=True)
    st.metric("Monthly revenue lost", lib.money(f.loc[f['Churn'] == 1, 'MonthlyCharges'].sum()), border=True)

left, right = st.columns(2)
with left:
    with st.container(border=True):
        attrs = ["Contract", "PaymentMethod", "InternetService", "OnlineSecurity", "TechSupport", "OnlineBackup",
                 "DeviceProtection", "StreamingTV", "StreamingMovies", "PaperlessBilling", "Partner", "Dependents",
                 "SeniorCitizen", "MultipleLines", "PhoneService", "gender"]
        attr = st.selectbox("Churn rate by attribute", attrs)
        st.altair_chart(lib.rate_bar(f, attr, overall, height=60 + 38 * f[attr].nunique()))
        st.caption("Vertical rule = churn rate of the current slice.")
with right:
    with st.container(border=True):
        num = st.segmented_control("Distribution by churn outcome", ["tenure", "MonthlyCharges", "TotalCharges"],
                                   default="tenure")
        chart = alt.Chart(f).transform_density(
            num, groupby=["Churn label"], as_=[num, "density"], extent=[0, float(df[num].max())]
        ).mark_area(opacity=0.45, line={"strokeWidth": 2}).encode(
            x=alt.X(f"{num}:Q"), y=alt.Y("density:Q", title="Density", axis=alt.Axis(labels=False)),
            color=alt.Color("Churn label:N", scale=lib.CHURN_SCALE, legend=alt.Legend(orient="top", title=None)),
            tooltip=["Churn label:N"],
        ).properties(height=300)
        st.altair_chart(chart)

with st.container(border=True):
    st.subheader("Tenure vs monthly charges")
    sample = f.sample(min(len(f), 4000), random_state=0)
    chart = alt.Chart(sample).mark_circle(size=22, opacity=0.5).encode(
        x=alt.X("tenure:Q", title="Tenure (months)"),
        y=alt.Y("MonthlyCharges:Q", title="Monthly charges ($)"),
        color=alt.Color("Churn label:N", scale=lib.CHURN_SCALE, legend=alt.Legend(orient="top", title=None)),
        tooltip=["customerID", "tenure", "MonthlyCharges", "Contract", "InternetService", "Churn label"],
    ).properties(height=380).interactive()
    st.altair_chart(chart)
    st.caption("New customers paying $70+ per month (top-left) are the densest churn region.")

st.subheader("Statistical evidence", icon=":material/science:")
c1, c2 = st.columns(2)
with c1:
    t = lib.table("stats_categorical_tests")[["feature", "cramers_v", "p_bonferroni", "highest_churn_level",
                                              "highest_churn_rate", "significant"]]
    st.dataframe(t, hide_index=True, column_config={
        "feature": "Attribute",
        "cramers_v": st.column_config.ProgressColumn("Cramer's V", min_value=0, max_value=0.5, format="%.3f"),
        "p_bonferroni": st.column_config.NumberColumn("p (Bonferroni)", format="%.1e"),
        "highest_churn_level": "Riskiest level",
        "highest_churn_rate": st.column_config.NumberColumn("Its churn rate", format="percent"),
        "significant": "Significant",
    })
    st.caption("Chi-square test of independence with bias-corrected Cramer's V (full dataset).")
with c2:
    o = lib.table("stats_logit_odds_ratios")
    st.dataframe(o, hide_index=True, column_config={
        "term": "Term",
        "odds_ratio": st.column_config.NumberColumn("Odds ratio", format="%.2f"),
        "ci_low": st.column_config.NumberColumn("95% CI low", format="%.2f"),
        "ci_high": st.column_config.NumberColumn("95% CI high", format="%.2f"),
        "p_value": st.column_config.NumberColumn("p-value", format="%.1e"),
    })
    st.caption("Adjusted odds ratios from an inferential logistic regression (>1 raises churn odds).")
