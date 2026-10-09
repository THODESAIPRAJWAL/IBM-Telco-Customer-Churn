import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

import lib

lib.require_artifacts()
m = lib.metrics()
md = m["metadata"]
sel = md["model_name"]
preds = lib.table("test_predictions")
model_names = [c for c in preds.columns if c not in ("customerID", "y", "Baseline (majority)")]
y = preds["y"].to_numpy()

st.caption(
    "Six model families were tuned with Optuna on identical 5-fold CV splits and combined into two ensembles. Selection used CV only: models within one standard error of the best CV AUC count "
    "as tied, and the best-calibrated of those is deployed. "
    "Everything on this page is computed on the untouched 20% hold-out set."
)

cv = pd.DataFrame(m["modelling"]["cv"])
test = pd.DataFrame(m["modelling"]["test"])
board = cv.merge(test[["model", "roc_auc", "auc_ci_low", "auc_ci_high", "pr_auc", "brier", "lift_top10"]], on="model")
board = board.sort_values("cv_auc_mean", ascending=False)
board.insert(0, "Deployed", board["model"] == sel)
st.dataframe(
    board[["Deployed", "model", "cv_auc_mean", "cv_auc_std", "roc_auc", "auc_ci_low", "auc_ci_high", "pr_auc",
           "brier", "lift_top10", "fit_seconds"]],
    hide_index=True,
    column_config={
        "Deployed": st.column_config.CheckboxColumn("Deployed", width="small"),
        "model": "Model",
        "cv_auc_mean": st.column_config.NumberColumn("CV ROC-AUC", format="%.4f"),
        "cv_auc_std": st.column_config.NumberColumn("CV sd", format="%.4f"),
        "roc_auc": st.column_config.NumberColumn("Test ROC-AUC", format="%.4f"),
        "auc_ci_low": st.column_config.NumberColumn("95% CI low", format="%.3f"),
        "auc_ci_high": st.column_config.NumberColumn("95% CI high", format="%.3f"),
        "pr_auc": st.column_config.NumberColumn("Test PR-AUC", format="%.4f"),
        "brier": st.column_config.NumberColumn("Brier", format="%.4f"),
        "lift_top10": st.column_config.NumberColumn("Top-10% lift", format="%.2fx"),
        "fit_seconds": st.column_config.NumberColumn("Fit time (s)", format="%.1f"),
    },
)
dl = pd.DataFrame(m["modelling"]["delong"])
lr = dl.set_index("model").loc["Logistic Regression"] if "Logistic Regression" in set(dl["model"]) else None
if lr is not None:
    verdict = "significant" if lr["p_value"] < 0.05 else "not statistically significant"
    st.caption(f"DeLong test, {sel} vs logistic regression: AUC difference {lr['diff']:+.4f}, "
               f"p = {lr['p_value']:.3f} ({verdict}). Tabular churn data is largely additive, so a well-engineered "
               f"linear model is a strong baseline.")

chosen = st.pills("Models to compare", model_names, default=[n for n in (sel, "Logistic Regression", "TabularNet (DL)")
                                                             if n in model_names], selection_mode="multi")
color = alt.Color("model:N", scale=alt.Scale(domain=list(lib.MODEL_COLORS), range=list(lib.MODEL_COLORS.values())),
                  legend=alt.Legend(orient="top", title=None))

c1, c2, c3 = st.columns(3)
roc_rows, pr_rows, cal_rows = [], [], []
for name in chosen:
    p = preds[name].to_numpy()
    fpr, tpr, _ = roc_curve(y, p)
    idx = np.linspace(0, len(fpr) - 1, min(len(fpr), 300)).astype(int)
    roc_rows.append(pd.DataFrame({"model": name, "fpr": fpr[idx], "tpr": tpr[idx]}))
    pr, rc, _ = precision_recall_curve(y, p)
    idx = np.linspace(0, len(pr) - 1, min(len(pr), 300)).astype(int)
    pr_rows.append(pd.DataFrame({"model": name, "recall": rc[idx], "precision": pr[idx]}))
    frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
    cal_rows.append(pd.DataFrame({"model": name, "predicted": mean, "observed": frac}))

if chosen:
    with c1, st.container(border=True):
        st.markdown("**ROC curve**")
        d = pd.concat(roc_rows)
        diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color=lib.GHOST, strokeWidth=1).encode(x="x", y="y")
        st.altair_chart((diag + alt.Chart(d).mark_line(strokeWidth=2).encode(
            x=alt.X("fpr:Q", title="False positive rate"), y=alt.Y("tpr:Q", title="True positive rate"),
            color=color, tooltip=["model", alt.Tooltip("fpr:Q", format=".2f"), alt.Tooltip("tpr:Q", format=".2f")],
        )).properties(height=300))
    with c2, st.container(border=True):
        st.markdown("**Precision-recall curve**")
        d = pd.concat(pr_rows)
        base = alt.Chart(pd.DataFrame({"y": [y.mean()]})).mark_rule(color=lib.GHOST).encode(y="y:Q")
        st.altair_chart((base + alt.Chart(d).mark_line(strokeWidth=2).encode(
            x=alt.X("recall:Q", title="Recall"), y=alt.Y("precision:Q", title="Precision", scale=alt.Scale(domain=[0, 1])),
            color=color, tooltip=["model", alt.Tooltip("recall:Q", format=".2f"), alt.Tooltip("precision:Q", format=".2f")],
        )).properties(height=300))
    with c3, st.container(border=True):
        st.markdown("**Calibration (reliability)**")
        d = pd.concat(cal_rows)
        diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color=lib.GHOST, strokeWidth=1).encode(x="x", y="y")
        st.altair_chart((diag + alt.Chart(d).mark_line(point=True, strokeWidth=2).encode(
            x=alt.X("predicted:Q", title="Mean predicted probability"),
            y=alt.Y("observed:Q", title="Observed churn rate"), color=color,
            tooltip=["model", alt.Tooltip("predicted:Q", format=".2f"), alt.Tooltip("observed:Q", format=".2f")],
        )).properties(height=300))

st.subheader("Threshold explorer", icon=":material/tune:")
st.caption(f"Profit-optimal threshold chosen on training out-of-fold predictions: **{md['threshold']:.3f}** "
           f"(F1-optimal would be {md['f1_threshold']:.3f}). Move the slider to see the trade-off for {sel}.")
thr = st.slider("Decision threshold", 0.05, 0.95, float(round(md["threshold"], 2)), 0.01)
p = preds[sel].to_numpy()
pred = p >= thr
tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
with st.container(horizontal=True):
    st.metric("Flagged as churn risk", f"{pred.sum():,}", f"{pred.mean():.0%} of customers", delta_color="off", delta_arrow="off", border=True)
    st.metric("Precision", f"{tp / max(tp + fp, 1):.1%}", border=True, help="Of flagged customers, share who churned")
    st.metric("Recall", f"{tp / max(tp + fn, 1):.1%}", border=True, help="Of churners, share who were flagged")
    st.metric("False alarms", f"{fp:,}", border=True)
    st.metric("Missed churners", f"{fn:,}", border=True)
cm = pd.DataFrame({"": ["Actually stayed", "Actually churned"], "Predicted stay": [tn, fn], "Predicted churn": [fp, tp]})
st.table(cm.set_index(""))

with st.expander("Class-imbalance experiment", icon=":material/balance:"):
    st.dataframe(pd.DataFrame(m["modelling"]["imbalance"]), hide_index=True)
    st.caption("Re-balancing (class weights, SMOTE, undersampling) leaves ranking quality unchanged but inflates "
               "probabilities and hurts calibration, so the natural distribution is kept and the decision threshold "
               "is set from business costs instead.")
with st.expander("Tuned hyper-parameters", icon=":material/settings:"):
    st.json(m["modelling"]["best_params"], expanded=False)
    st.caption(f"Stacked ensemble meta-weights on base-model log-odds: {m['modelling']['stack_weights']}")
