import altair as alt
import pandas as pd
import streamlit as st

import lib

lib.require_artifacts()
pred = lib.predictor()
bp = pred.business
L = lib.CATEGORY_LEVELS
df = lib.customers()

st.caption("Score any customer: churn probability, risk tier, the retention decision under current campaign "
           "economics, *why* the model thinks so, and which offers the model expects to reduce risk most.")


@st.cache_data(max_entries=256, show_spinner=False)
def explain(row: dict) -> pd.DataFrame:
    return pred.explain(pd.DataFrame([row]))


@st.cache_data(max_entries=256, show_spinner=False)
def actions(row: dict) -> pd.DataFrame:
    return pred.next_best_actions(pd.DataFrame([row]))


# ---- choose a starting profile ------------------------------------------------
examples = df.sort_values("customerID")
start = st.selectbox("Start from an existing customer (or edit the fields below)", examples["customerID"],
                     index=int(examples["customerID"].tolist().index("7590-VHVEG")))
base = examples.set_index("customerID").loc[start]

with st.form("customer"):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Account**")
        tenure = st.number_input("Tenure (months)", 0, 120, int(base["tenure"]))
        contract = st.selectbox("Contract", L["Contract"], index=L["Contract"].index(base["Contract"]))
        payment = st.selectbox("Payment method", L["PaymentMethod"], index=L["PaymentMethod"].index(base["PaymentMethod"]))
        paperless = st.segmented_control("Paperless billing", ["Yes", "No"], default=base["PaperlessBilling"])
        monthly = st.number_input("Monthly charges ($)", 0.0, 500.0, float(base["MonthlyCharges"]), step=5.0)
        total = st.number_input("Total charges ($)", 0.0, 20000.0, float(base["TotalCharges"]), step=50.0)
    with c2:
        st.markdown("**Customer**")
        senior = st.segmented_control("Senior citizen", ["Yes", "No"], default="Yes" if base["SeniorCitizen"] else "No")
        partner = st.segmented_control("Partner", ["Yes", "No"], default=base["Partner"])
        dependents = st.segmented_control("Dependents", ["Yes", "No"], default=base["Dependents"])
        st.markdown("**Phone**")
        phone = st.segmented_control("Phone service", ["Yes", "No"], default=base["PhoneService"])
        multi = st.segmented_control("Multiple lines", ["Yes", "No"],
                                     default=base["MultipleLines"] if base["MultipleLines"] != "No phone service" else "No")
    with c3:
        st.markdown("**Internet**")
        internet = st.selectbox("Internet service", L["InternetService"], index=L["InternetService"].index(base["InternetService"]))
        addon_vals = {}
        for svc, label in [("OnlineSecurity", "Online security"), ("OnlineBackup", "Online backup"),
                           ("DeviceProtection", "Device protection")]:
            addon_vals[svc] = st.toggle(label, value=base[svc] == "Yes")
    with c4:
        st.markdown("**Support & streaming**")
        for svc, label in [("TechSupport", "Tech support"), ("StreamingTV", "Streaming TV"),
                           ("StreamingMovies", "Streaming movies")]:
            addon_vals[svc] = st.toggle(label, value=base[svc] == "Yes")
    submitted = st.form_submit_button("Score customer", icon=":material/bolt:", type="primary")

row = {
    "customerID": start, "SeniorCitizen": 1 if senior == "Yes" else 0, "Partner": partner or "No",
    "Dependents": dependents or "No", "tenure": int(tenure), "PhoneService": phone or "No",
    "MultipleLines": ("Yes" if multi == "Yes" else "No") if phone == "Yes" else "No phone service",
    "InternetService": internet, "Contract": contract, "PaperlessBilling": paperless or "No",
    "PaymentMethod": payment, "MonthlyCharges": float(monthly), "TotalCharges": float(total),
}
for svc, on in addon_vals.items():
    row[svc] = "No internet service" if internet == "No" else ("Yes" if on else "No")

frame = pd.DataFrame([row])
s = pred.score(frame).iloc[0]
p = float(s["churn_probability"])
tier = s["risk_tier"]
badge_color, badge_icon = lib.TIER_BADGE[tier]

with st.container(horizontal=True):
    st.metric("Churn probability", f"{p:.1%}", border=True)
    with st.container(border=True):
        st.caption("Risk tier")
        st.badge(tier, icon=badge_icon, color=badge_color)
    st.metric("Margin at risk", lib.money(s["value_at_risk"]), f"{bp.clv_horizon_months} months x "
              f"{bp.gross_margin:.0%} margin", delta_color="off", delta_arrow="off", border=True)
    st.metric("Expected value of an offer", lib.money(s["expected_campaign_value"]),
              "p x success x value - cost", delta_color="off", delta_arrow="off", border=True)
    with st.container(border=True):
        st.caption("Decision")
        if s["target_for_retention"]:
            st.badge("Target with retention offer", icon=":material/campaign:", color="blue")
        else:
            st.badge("Do not target", icon=":material/do_not_disturb_on:", color="gray")

left, right = st.columns([1.25, 1])
with left, st.container(border=True):
    st.markdown("**Why this score?** SHAP contributions in percentage points")
    with st.spinner("Computing SHAP explanation..."):
        ex = explain(row)
    top = ex.head(10).copy()
    rest = ex["contribution"].iloc[10:].sum()
    top["label"] = top["feature"] + " = " + top["value"].astype(str)
    if abs(rest) > 1e-9:
        top = pd.concat([top, pd.DataFrame([{"label": f"{len(ex) - 10} other attributes", "contribution": rest}])])
    top["pp"] = top["contribution"] * 100
    top["direction"] = top["pp"].apply(lambda v: "Raises churn risk" if v > 0 else "Lowers churn risk")
    chart = alt.Chart(top).mark_bar(cornerRadiusEnd=4, height={"band": 0.65}).encode(
        y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=320)),
        x=alt.X("pp:Q", title="Contribution (pp)",
                scale=alt.Scale(domain=[min(top["pp"].min() * 1.3, -3), max(top["pp"].max() * 1.2, 3)])),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["Raises churn risk", "Lowers churn risk"],
                                                       range=[lib.CHURNED, lib.RETAINED]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["label", alt.Tooltip("pp:Q", format="+.1f", title="Contribution (pp)")],
    )
    text_pos = chart.mark_text(align="left", dx=4, fontSize=11).encode(
        text=alt.Text("pp:Q", format="+.1f"), color=alt.value("#52514e")).transform_filter("datum.pp >= 0")
    text_neg = chart.mark_text(align="right", dx=-4, fontSize=11).encode(
        text=alt.Text("pp:Q", format="+.1f"), color=alt.value("#52514e")).transform_filter("datum.pp < 0")
    st.altair_chart((chart + text_pos + text_neg).properties(height=380))
    st.caption(f"Baseline (average customer) {ex['base_value'].iloc[0]:.1%} + contributions = {p:.1%}. "
               "Tenure and TotalCharges are strongly correlated, so SHAP splits credit between them - read them together.")
with right, st.container(border=True):
    st.markdown("**Next-best action** - simulated effect of each offer")
    nba = actions(row)
    if nba.empty:
        st.caption("No applicable actions for this profile.")
    else:
        nba["new_p"] = nba["new_probability"]
        st.dataframe(nba[["action", "new_p", "risk_reduction"]], hide_index=True, column_config={
            "action": st.column_config.TextColumn("Action", width="large"),
            "new_p": st.column_config.NumberColumn("New risk", format="percent"),
            "risk_reduction": st.column_config.ProgressColumn("Risk reduction", min_value=0.0,
                                                              max_value=max(0.3, float(nba["risk_reduction"].max())),
                                                              format="percent"),
        })
    st.caption(":material/info: Model-implied (associational) effects - use them to prioritise which offers to "
               "A/B test, not as proven causal lift.")
