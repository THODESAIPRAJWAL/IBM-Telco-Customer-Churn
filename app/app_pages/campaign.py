import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import lib

lib.require_artifacts()
m = lib.metrics()
sel = lib.selected_model_name()
default = lib.business_from_metadata()
preds = lib.table("test_predictions").merge(lib.customers()[["customerID", "MonthlyCharges"]], on="customerID")
scale = m["business"]["scale_to_full_base"]

st.caption("Plug in your own campaign economics. The planner re-optimises who to contact using the model's "
           "hold-out predictions and projects the result to the full customer base.")

with st.container(border=True):
    c1, c2, c3, c4, c5 = st.columns(5)
    horizon = c1.slider("Value horizon (months)", 6, 48, default.clv_horizon_months, 6,
                        help="How many months of revenue a saved customer is worth")
    margin = c2.slider("Gross margin", 0.2, 0.9, default.gross_margin, 0.05, format="%.2f")
    contact = c3.slider("Contact cost ($)", 0.0, 50.0, default.contact_cost, 1.0)
    incentive = c4.slider("Incentive cost ($)", 0.0, 200.0, default.incentive_cost, 5.0)
    success = c5.slider("Offer success rate", 0.05, 0.9, default.offer_success_rate, 0.05, format="%.2f",
                        help="Share of targeted would-be churners who stay because of the offer")

bp = lib.BusinessParams(horizon, margin, contact, incentive, success)
y = preds["y"].to_numpy()
p = preds[sel].to_numpy()
value = preds["MonthlyCharges"].to_numpy() * horizon * margin

curve = lib.B.profit_by_threshold(y, p, value, bp)
best = curve.loc[curve["net_profit"].idxmax()]
policies = lib.B.compare_policies(y, p, value, float(best["threshold"]), bp)
ev = policies.loc["Value-aware rule (p*s*v > c)"]
everyone = policies.loc["Target everyone"]
oracle = policies.loc["Oracle (perfect foresight)"]

with st.container(horizontal=True):
    st.metric("Projected net profit", lib.money(ev["net_profit"] * scale), f"ROI {ev['roi']:.0%}" if ev["campaign_cost"] else None,
              delta_color="off", delta_arrow="off", border=True,
              help="Value-aware rule: contact when p x success x value > cost")
    st.metric("Customers to contact", f"{int(ev['n_targeted'] * scale):,}", f"{ev['share_targeted']:.0%} of base",
              delta_color="off", delta_arrow="off", border=True)
    st.metric("Customers saved (expected)", f"{ev['expected_customers_saved'] * scale:,.0f}", border=True)
    st.metric("vs contacting everyone", lib.money(everyone["net_profit"] * scale), border=True)
    st.metric("Share of perfect-foresight profit", f"{ev['net_profit'] / oracle['net_profit']:.0%}" if oracle["net_profit"] > 0 else "-",
              border=True)

with st.container(border=True):
    st.markdown("**Profit curve** - net profit by how deep the campaign goes")
    rng = np.random.default_rng(0)
    rows = []
    for label, score in [("Rank by churn probability", p), ("Rank by expected value (p x value)", p * value),
                         ("Random targeting", rng.random(len(p)))]:
        d = lib.B.profit_by_depth(y, score, value, bp)
        idx = np.linspace(0, len(d) - 1, 250).astype(int)
        rows.append(d.iloc[idx].assign(strategy=label, profit=lambda x: x["profit"] * scale))
    d = pd.concat(rows)
    chart = alt.Chart(d).mark_line(strokeWidth=2).encode(
        x=alt.X("share:Q", title="Share of customers contacted", axis=alt.Axis(format="%")),
        y=alt.Y("profit:Q", title="Net profit ($, full base)", axis=alt.Axis(format="$,.0f")),
        color=alt.Color("strategy:N", scale=alt.Scale(
            domain=["Rank by churn probability", "Rank by expected value (p x value)", "Random targeting"],
            range=[lib.RETAINED, "#1baf7a", lib.GHOST]), legend=alt.Legend(orient="top", title=None, labelLimit=320)),
        tooltip=["strategy", alt.Tooltip("share:Q", format=".0%", title="Contacted"),
                 alt.Tooltip("profit:Q", format="$,.0f", title="Net profit")],
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#52514e", strokeWidth=1).encode(y="y:Q")
    st.altair_chart((zero + chart).properties(height=360))
    st.caption(f"Best uniform threshold under these assumptions: {best['threshold']:.2f} "
               f"(contacts {best['share_targeted']:.0%} of customers).")

show = policies.copy()
for c in ["n_targeted", "churners_reached", "expected_customers_saved", "retained_value", "campaign_cost", "net_profit"]:
    show[c] = show[c].astype(float) * scale
st.dataframe(show.reset_index()[["policy", "n_targeted", "churners_reached", "expected_customers_saved",
                                 "campaign_cost", "retained_value", "net_profit", "roi"]], hide_index=True,
             column_config={
                 "policy": st.column_config.TextColumn("Policy", width="large"),
                 "n_targeted": st.column_config.NumberColumn("Contacted", format="%,.0f"),
                 "churners_reached": st.column_config.NumberColumn("Churners reached", format="%,.0f"),
                 "expected_customers_saved": st.column_config.NumberColumn("Saved (exp.)", format="%,.0f"),
                 "campaign_cost": st.column_config.NumberColumn("Cost", format="$%,.0f"),
                 "retained_value": st.column_config.NumberColumn("Margin retained", format="$%,.0f"),
                 "net_profit": st.column_config.NumberColumn("Net profit", format="$%,.0f"),
                 "roi": st.column_config.NumberColumn("ROI", format="percent"),
             })
st.caption("Hold-out outcomes scaled to the full base. Assumes the offer only changes behaviour of would-be churners; "
           "validate with a randomised control group before rollout.")
