import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import lib

lib.require_artifacts()
sel = lib.selected_model_name()
imp = lib.table("shap_global_importance")
perm = lib.table("permutation_importance")
sv = lib.table("shap_values_sample")

st.caption(
    f"SHAP values are computed on the deployed **{sel}** in the space of the original customer attributes, so each "
    "number reads as *percentage points of churn probability* - e.g. \"Contract = Month-to-month adds +12 pp\"."
)

c1, c2 = st.columns(2)
with c1, st.container(border=True):
    st.markdown("**Global importance - mean |SHAP|**")
    d = imp.assign(pp=imp["mean_abs_shap"] * 100)
    st.altair_chart(alt.Chart(d).mark_bar(color=lib.RETAINED, cornerRadiusEnd=4, height={"band": 0.65}).encode(
        y=alt.Y("feature:N", sort="-x", title=None),
        x=alt.X("pp:Q", title="Mean |SHAP| (pp of churn probability)"),
        tooltip=["feature", alt.Tooltip("pp:Q", format=".2f", title="Mean |SHAP| (pp)")],
    ).properties(height=440))
with c2, st.container(border=True):
    st.markdown("**Permutation importance - hold-out AUC drop**")
    st.altair_chart(alt.Chart(perm).mark_bar(color=lib.RETAINED, cornerRadiusEnd=4, height={"band": 0.65}).encode(
        y=alt.Y("feature:N", sort="-x", title=None),
        x=alt.X("auc_drop:Q", title="Drop in test ROC-AUC when shuffled"),
        tooltip=["feature", alt.Tooltip("auc_drop:Q", format=".4f"), alt.Tooltip("std:Q", format=".4f")],
    ).properties(height=440))

st.subheader("How each attribute value moves the prediction", icon=":material/insights:")
feature = st.selectbox("Attribute", imp["feature"].tolist())
d = pd.DataFrame({"value": sv[feature], "shap_pp": sv[f"shap_{feature}"] * 100})
with st.container(border=True):
    if feature in lib.CATEGORY_LEVELS or feature == "SeniorCitizen":
        d["value"] = d["value"].astype(str)
        d["direction"] = np.where(d["shap_pp"] > 0, "Raises churn risk", "Lowers churn risk")
        dots = alt.Chart(d).mark_circle(size=40, opacity=0.45).encode(
            y=alt.Y("value:N", title=None), yOffset=alt.YOffset("jitter:Q"),
            x=alt.X("shap_pp:Q", title="SHAP value (pp of churn probability)"),
            color=alt.Color("direction:N", scale=alt.Scale(domain=["Raises churn risk", "Lowers churn risk"],
                                                           range=[lib.CHURNED, lib.RETAINED]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["value", alt.Tooltip("shap_pp:Q", format="+.1f", title="SHAP (pp)")],
        ).transform_calculate(jitter="random()")
        means = alt.Chart(d).mark_tick(thickness=3, size=34, color="#0b0b0b").encode(
            y="value:N", x="mean(shap_pp):Q", tooltip=[alt.Tooltip("mean(shap_pp):Q", format="+.1f", title="Mean (pp)")])
        st.altair_chart((dots + means).properties(height=90 + 70 * d["value"].nunique()))
        st.caption("Each dot is a hold-out customer; the black tick is the mean effect of that value.")
    else:
        st.altair_chart(alt.Chart(d).mark_circle(size=30, opacity=0.55, color=lib.RETAINED).encode(
            x=alt.X("value:Q", title=feature), y=alt.Y("shap_pp:Q", title="SHAP value (pp)"),
            tooltip=[alt.Tooltip("value:Q", title=feature), alt.Tooltip("shap_pp:Q", format="+.1f", title="SHAP (pp)")],
        ).properties(height=380))

with st.expander("TreeSHAP on engineered features (model diagnostics)", icon=":material/account_tree:"):
    st.image(str(lib.FIGURES / "shap_tree_beeswarm.png"))
    st.caption("Exact TreeSHAP on the best gradient-boosting model, showing how the engineered features "
               "(e.g. charge_delta, fiber_no_support, num_protection) contribute.")
