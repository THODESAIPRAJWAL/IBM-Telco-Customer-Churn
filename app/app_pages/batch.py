import pandas as pd
import streamlit as st

import lib

lib.require_artifacts()
pred = lib.predictor()

from churn.data import clean, validate  # noqa: E402  (lib puts src/ on the path)

st.caption("Upload a CSV in the IBM Telco schema (the `Churn` column is optional) and download scores, risk tiers "
           "and retention decisions for every customer.")

template = lib.customers().drop(columns=["Churn", "Churn label"]).head(25)
st.download_button("Download a template CSV (25 sample customers)", template.to_csv(index=False),
                   "churn_scoring_template.csv", "text/csv", icon=":material/download:")

up = st.file_uploader("Customer file", type=["csv"])
if up is None:
    st.stop()

raw = pd.read_csv(up)
try:
    df = clean(raw)
except Exception as exc:  # noqa: BLE001 - surface any parsing problem to the user
    st.error(f"Could not parse the file: {exc}", icon=":material/error:")
    st.stop()
problems = validate(df, require_target=False)
if problems:
    st.error("The file does not match the expected schema:\n\n- " + "\n- ".join(problems), icon=":material/error:")
    st.stop()

scores = pred.score(df)
out = pd.concat([scores, df.drop(columns=["customerID"], errors="ignore")], axis=1)
with st.container(horizontal=True):
    st.metric("Customers scored", f"{len(out):,}", border=True)
    st.metric("Mean churn risk", f"{out['churn_probability'].mean():.1%}", border=True)
    st.metric("Recommended for retention offer", f"{int(out['target_for_retention'].sum()):,}", border=True)
    st.metric("Expected campaign value", lib.money(out.loc[out["target_for_retention"], "expected_campaign_value"].sum()),
              border=True)
st.dataframe(out.sort_values("churn_probability", ascending=False), hide_index=True, column_config={
    "churn_probability": st.column_config.ProgressColumn("Churn risk", min_value=0, max_value=1, format="percent"),
    "value_at_risk": st.column_config.NumberColumn("Margin at risk", format="$%.0f"),
    "expected_campaign_value": st.column_config.NumberColumn("Offer EV", format="$%.0f"),
})
st.download_button("Download scored file", out.to_csv(index=False), "churn_scores.csv", "text/csv",
                   icon=":material/download:", type="primary")
