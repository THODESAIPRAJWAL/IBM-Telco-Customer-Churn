import altair as alt
import pandas as pd
import streamlit as st

import lib

lib.require_artifacts()
m = lib.metrics()
thr = m["metadata"]["threshold"]
scored = lib.scored_customers()
summary = lib.table("segment_summary")
personas = lib.table("persona_profiles")

st.caption("Every customer is scored out-of-sample (cross-validated for training customers, hold-out predictions "
           "for test customers), then placed on a risk x value grid with a playbook per segment.")

order = ["Priority save", "Low-cost save", "Protect & grow", "Maintain"]
with st.container(horizontal=True):
    for seg in order:
        r = summary.set_index("segment").loc[seg]
        st.metric(seg, f"{int(r['customers']):,} customers", f"{r['share_of_expected_loss']:.0%} of expected loss",
                  delta_color="off", delta_arrow="off", border=True, help=r["playbook"])

left, right = st.columns([1.35, 1])
with left, st.container(border=True):
    st.markdown("**Risk x value matrix**")
    d = scored.sample(min(len(scored), 5000), random_state=0)
    d = d.assign(outcome=d["Churn"].map({1: "Churned", 0: "Retained"}))
    pts = alt.Chart(d).mark_circle(size=18, opacity=0.45).encode(
        x=alt.X("churn_probability:Q", title="Predicted churn probability", axis=alt.Axis(format="%")),
        y=alt.Y("value_at_risk:Q", title="Margin at risk ($)"),
        color=alt.Color("outcome:N", scale=lib.CHURN_SCALE, legend=alt.Legend(orient="top", title="Actual outcome")),
        tooltip=["customerID", "segment", alt.Tooltip("churn_probability:Q", format=".1%"),
                 alt.Tooltip("value_at_risk:Q", format="$,.0f"), "Contract", "tenure", "outcome"],
    )
    vrule = alt.Chart(pd.DataFrame({"x": [thr]})).mark_rule(color="#52514e").encode(x="x:Q")
    hrule = alt.Chart(pd.DataFrame({"y": [scored["value_at_risk"].median()]})).mark_rule(color="#52514e").encode(y="y:Q")
    st.altair_chart((pts + vrule + hrule).properties(height=420).interactive())
    st.caption(f"Vertical rule: profit-optimal risk threshold ({thr:.2f}); horizontal rule: median margin at risk.")
with right, st.container(border=True):
    st.markdown("**Segment playbook**")
    st.dataframe(summary[["segment", "customers", "actual_churn_rate", "avg_monthly_charges", "playbook"]],
                 hide_index=True, column_config={
                     "segment": "Segment", "customers": st.column_config.NumberColumn("Customers", format="%,d"),
                     "actual_churn_rate": st.column_config.NumberColumn("Churn rate", format="percent"),
                     "avg_monthly_charges": st.column_config.NumberColumn("Avg $/mo", format="$%.0f"),
                     "playbook": st.column_config.TextColumn("Playbook", width="large")})
    st.markdown("**Behavioural personas** (K-Means)")
    st.dataframe(personas[["persona", "customers", "Churn", "MonthlyCharges", "tenure"]], hide_index=True,
                 column_config={"persona": st.column_config.TextColumn("Persona", width="medium"),
                                "customers": st.column_config.NumberColumn("Customers", format="%,d"),
                                "Churn": st.column_config.ProgressColumn("Churn rate", min_value=0, max_value=1, format="percent"),
                                "MonthlyCharges": st.column_config.NumberColumn("Avg $/mo", format="$%.0f"),
                                "tenure": st.column_config.NumberColumn("Avg tenure", format="%.0f mo")})

st.subheader("Target list", icon=":material/list_alt:")
pick = st.pills("Segments", order, default=["Priority save"], selection_mode="multi")
tl = scored[scored["segment"].isin(pick or order)].sort_values("expected_loss", ascending=False)
cols = ["customerID", "segment", "persona", "churn_probability", "value_at_risk", "expected_loss", "Contract",
        "tenure", "MonthlyCharges", "InternetService", "PaymentMethod"]
st.dataframe(tl[cols], hide_index=True, column_config={
    "churn_probability": st.column_config.ProgressColumn("Churn risk", min_value=0, max_value=1, format="percent"),
    "value_at_risk": st.column_config.NumberColumn("Margin at risk", format="$%.0f"),
    "expected_loss": st.column_config.NumberColumn("Expected loss", format="$%.0f"),
    "MonthlyCharges": st.column_config.NumberColumn("$/mo", format="$%.2f"),
})
st.download_button("Download target list (CSV)", tl[cols].to_csv(index=False), "retention_target_list.csv",
                   "text/csv", icon=":material/download:")
