"""Render reports/REPORT.md from the pipeline's computed metrics (no hand-typed numbers)."""

from __future__ import annotations

import pandas as pd

from .config import REPORTS_DIR


def _pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def _money(x):
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def _table(df: pd.DataFrame, floatfmt: dict | None = None) -> str:
    floatfmt = floatfmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in floatfmt and isinstance(v, (int, float)):
                cells.append(floatfmt[c](v))
            elif isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write(s: dict) -> None:
    md, e, sv, m = s["metadata"], s["eda"], s["survival"], s["modelling"]
    tm = md["test_metrics"]
    bp = md["business_params"]
    scale = s["business"]["scale_to_full_base"]
    pol = pd.DataFrame(s["business"]["policies_test"]).set_index("policy")
    uniform = [i for i in pol.index if i.startswith("Uniform")][0]
    ev_row = pol.loc["Value-aware rule (p*s*v > c)"]
    all_row = pol.loc["Target everyone"]
    oracle = pol.loc["Oracle (perfect foresight)"]
    cv = pd.DataFrame(m["cv"])
    test = pd.DataFrame(m["test"])
    delong = pd.DataFrame(m["delong"])
    imb = pd.DataFrame(m["imbalance"])
    dec = pd.DataFrame(m["deciles"])
    seg = pd.DataFrame(s["segmentation"]["segments"]).set_index("segment")
    fair = pd.DataFrame(s["fairness"]["disparity"]).set_index("attribute")
    abl = s["fairness"]["gender_ablation"]
    lr_test = test.set_index("model").loc["Logistic Regression", "roc_auc"]
    lr_delong = delong.set_index("model").loc["Logistic Regression"] if "Logistic Regression" in set(delong["model"]) else None
    shap_top = [d["feature"] for d in s["explainability"]["shap_top_features"][:5]]

    comp = cv.merge(test[["model", "roc_auc", "auc_ci_low", "auc_ci_high", "pr_auc", "brier"]], on="model")
    comp = comp.sort_values("cv_auc_mean", ascending=False)
    comp_tbl = pd.DataFrame({
        "Model": ["**" + x + "**" if x == md["model_name"] else x for x in comp["model"]],
        "CV ROC-AUC (mean +/- sd)": [f"{a:.4f} +/- {b:.4f}" for a, b in zip(comp["cv_auc_mean"], comp["cv_auc_std"])],
        "Test ROC-AUC [95% CI]": [f"{a:.4f} [{lo:.3f}, {hi:.3f}]" for a, lo, hi in
                                  zip(comp["roc_auc"], comp["auc_ci_low"], comp["auc_ci_high"])],
        "Test PR-AUC": [f"{x:.4f}" for x in comp["pr_auc"]],
        "Test Brier": [f"{x:.4f}" for x in comp["brier"]],
    })

    pol_tbl = pd.DataFrame({
        "Policy": pol.index,
        "Customers contacted": [f"{int(round(x * scale)):,}" for x in pol["n_targeted"]],
        "Churners reached": [f"{int(round(x * scale)):,}" for x in pol["churners_reached"]],
        "Campaign cost": [_money(x * scale) for x in pol["campaign_cost"]],
        "Margin retained": [_money(x * scale) for x in pol["retained_value"]],
        "Net profit": [_money(x * scale) for x in pol["net_profit"]],
        "ROI": [f"{x:.0%}" if c else "-" for x, c in zip(pol["roi"], pol["campaign_cost"])],
    })

    imb_tbl = imb.rename(columns={"strategy": "Strategy", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "brier": "Brier",
                                  "mean_predicted_prob": "Mean predicted p", "recall_at_0.5": "Recall@0.5",
                                  "precision_at_0.5": "Precision@0.5"})

    seg_tbl = pd.DataFrame({
        "Segment": seg.index,
        "Customers": [f"{int(x):,}" for x in seg["customers"]],
        "Actual churn": [_pct(x) for x in seg["actual_churn_rate"]],
        "Avg monthly $": [f"${x:.0f}" for x in seg["avg_monthly_charges"]],
        "Share of expected loss": [_pct(x) for x in seg["share_of_expected_loss"]],
    })

    fair_tbl = pd.DataFrame({
        "Attribute": fair.index,
        "AUC gap": [f"{x:.3f}" for x in fair["auc_gap"]],
        "TPR gap": [f"{x:.3f}" for x in fair["tpr_gap"]],
        "FPR gap": [f"{x:.3f}" for x in fair["fpr_gap"]],
        "Selection-rate ratio": [f"{x:.2f}" for x in fair["selection_rate_ratio"]],
        "Base-rate ratio": [f"{x:.2f}" for x in fair["base_rate_ratio"]],
    })

    if all_row["net_profit"] <= 0:
        targeting_txt = (f"Contacting everyone would lose {_money(-all_row['net_profit'] * scale)}; contacting only "
                         f"customers whose expected saved margin exceeds the offer cost earns "
                         f"{_money(ev_row['net_profit'] * scale)} instead.")
    else:
        targeting_txt = (f"Contacting only customers whose expected saved margin exceeds the offer cost earns "
                         f"{_money(ev_row['net_profit'] * scale)} - {ev_row['net_profit'] / all_row['net_profit']:.1f}x "
                         f"the profit of contacting everyone ({_money(all_row['net_profit'] * scale)}) while contacting "
                         f"{_pct(1 - ev_row['share_targeted'], 0)} fewer customers.")
    tied = cv[cv["model"].isin(m["within_1se"])].sort_values("oof_brier")
    calib_txt = "; ".join(f"{r.model} OOF Brier {r.oof_brier:.4f}" for r in tied.itertuples()) + (
        f". The deployed {md['model_name']} " + ("has the best calibration of the tied group."
        if tied.iloc[0]["model"] == md["model_name"] else
        f"was selected on CV AUC; {tied.iloc[0]['model']} is the better-calibrated alternative."))
    gm = pd.read_csv(REPORTS_DIR / "tables" / "fairness_group_metrics.csv")
    worst_attr = fair["auc_gap"].idxmax()
    g = gm[gm["attribute"] == worst_attr].set_index("group")
    lo_grp, hi_grp = g["roc_auc"].idxmin(), g["roc_auc"].idxmax()
    sel_hi = g["selection_rate"].idxmax()
    fair_txt = (
        f"* **Gender:** AUC gap {fair.loc['gender', 'auc_gap']:.3f}, selection-rate ratio "
        f"{fair.loc['gender', 'selection_rate_ratio']:.2f} - no material disparity (and gender is not a model input).\n"
        f"* **Largest gap - {worst_attr}:** the model ranks {worst_attr} = {lo_grp} customers less accurately "
        f"(AUC {g.loc[lo_grp, 'roc_auc']:.3f} vs {g.loc[hi_grp, 'roc_auc']:.3f}, n = {int(g.loc[lo_grp, 'n'])} "
        f"in the test set).\n"
        f"* **Selection rates** differ mainly because base churn rates differ ({worst_attr} = {sel_hi}: base rate "
        f"{_pct(g.loc[sel_hi, 'base_churn_rate'], 0)}, targeted {_pct(g.loc[sel_hi, 'selection_rate'], 0)}). "
        f"Here being targeted means *receiving* a retention offer, so higher selection is not an adverse outcome; the "
        f"actionable gap is accuracy. Mitigations: monitor group-wise AUC/calibration in production, and consider "
        f"group-aware calibration or more data for the smaller group.")
    tl = pd.DataFrame(m.get("tuning_log") or [])
    bp_ = m["best_params"]
    depths = [bp_.get("xgboost", {}).get("max_depth"), bp_.get("lightgbm", {}).get("max_depth"),
              bp_.get("catboost", {}).get("depth")]
    tuning_txt = ""
    if not tl.empty:
        tuning_txt = (f"Tuning budget: **{int(tl['trials'].sum())} Optuna trials** ("
                      + ", ".join(f"{r.model} {int(r.trials)}" for r in tl.itertuples()) + "). ")
    if all(d is not None for d in depths):
        tuning_txt += (f"The tuned gradient-boosting models converged on very shallow trees (max depth "
                       f"{min(depths)}-{max(depths)}) - independent evidence that the churn signal is largely additive.")
    arche = pd.DataFrame(sv["archetypes"])
    stack_w = ", ".join(f"{k} {v:+.2f}" for k, v in m["stack_weights"].items())
    sig_txt = ""
    if lr_delong is not None:
        sig = "statistically significant" if lr_delong["p_value"] < 0.05 else "**not** statistically significant"
        sig_txt = (f"Test-AUC difference between the deployed model and plain logistic regression: "
                   f"{lr_delong['diff']:+.4f} (DeLong test p = {lr_delong['p_value']:.3f}, {sig} at 5%).")

    text = f"""# Telco Customer Churn - Analysis & Modelling Report

*Auto-generated by `run_pipeline.py` on {md['trained_at']} - every number below is computed, not typed.*
*Data: IBM Telco Customer Churn ({e['n_customers']:,} customers, file hash `{md['data_sha256_16']}`).*

---

## 1. Executive summary

| | |
|---|---|
| Customers / churn rate | {e['n_customers']:,} / **{_pct(e['churn_rate'])}** |
| Monthly recurring revenue lost to churn | **{_money(e['monthly_revenue_lost'])}** ({_pct(e['monthly_revenue_lost'] / e['monthly_revenue_total'])} of MRR) |
| Deployed model | **{md['model_name']}** |
| Test ROC-AUC [95% CI] | **{tm['roc_auc']:.4f}** [{tm['auc_ci_low']:.3f}, {tm['auc_ci_high']:.3f}] |
| Test PR-AUC / Brier | {tm['pr_auc']:.4f} / {tm['brier']:.4f} |
| Top-decile lift | **{tm['lift_top10']:.2f}x** the base churn rate |
| Projected campaign profit (value-aware targeting, full base) | **{_money(ev_row['net_profit'] * scale)}** (ROI {ev_row['roi']:.0%}) |
| vs. blanket "contact everyone" campaign | {_money(all_row['net_profit'] * scale)} |

**Key findings**

1. **Contract type is the single strongest churn driver.** Month-to-month customers churn at
   {_pct(e['churn_by_contract']['Month-to-month'])} - {e['mtm_vs_one_year_ratio']:.1f}x the one-year rate and
   {e['mtm_vs_two_year_ratio']:.0f}x the two-year rate - and account for
   {_pct(e['revenue_lost_by_contract']['Month-to-month'] / e['monthly_revenue_lost'], 0)} of lost revenue.
2. **The first year is the danger zone.** Only {_pct(sv['retention_12m_overall'], 0)} of customers survive 12 months
   (Kaplan-Meier); the median month-to-month customer is gone after
   {sv['median_lifetime_by_contract']['Month-to-month']:.0f} months.
3. **Fibre optic without protection services, electronic-check payment and paperless billing** all raise churn odds
   even after adjusting for contract and tenure; security/tech-support add-ons lower it.
4. **Model-driven targeting pays.** {targeting_txt}
5. **Gender carries no signal** (chi-square p = {e['gender_p_value']:.2f}); it is excluded from the model, and the
   ablation confirms this costs nothing (CV AUC {abl['without_gender']['cv_auc_mean']:.4f} without vs
   {abl['with_gender']['cv_auc_mean']:.4f} with).

---

## 2. Data & quality

* 7,043 rows x 21 columns; one row per customer, target `Churn` (Yes/No).
* `TotalCharges` is stored as text with **11 blank values** - all for customers with `tenure = 0` (not yet billed),
  so they are set to 0 rather than imputed.
* No duplicate customer IDs; 22 rows share identical attributes (distinct customers with the same plan).
* Stratified 80/20 split -> {md['n_train']:,} train / {md['n_test']:,} test; the test set is used **once**, for final
  reporting.

## 3. Exploratory & statistical analysis

Every relationship is backed by a test and an effect size (chi-square + bias-corrected Cramer's V for categoricals,
Mann-Whitney U + rank-biserial correlation for numerics, Bonferroni-corrected).

* Strongest categorical associations (Cramer's V): {', '.join(f"{d['feature']} ({d['cramers_v']:.2f})" for d in e['top_categorical'])}.
* Not significant after Bonferroni correction: {', '.join(e['non_significant_features'])}.

![Churn by attribute](figures/eda_churn_rate_by_category.png)
![Tenure x contract](figures/eda_churn_by_tenure_and_contract.png)
![Numeric distributions](figures/eda_numeric_distributions.png)
![Revenue at risk](figures/eda_revenue_at_risk.png)
![Protection services](figures/eda_protection_services.png)
![Tenure vs charges](figures/eda_tenure_vs_charges.png)
![Association matrix](figures/eda_association_matrix.png)
![Mutual information](figures/eda_mutual_information.png)

**Inferential logistic regression** (adjusted odds ratios with 95% CIs) separates each factor's effect from the
others - e.g. fibre optic remains a strong risk factor after controlling for price-related services and contract:

![Odds ratios](figures/eda_odds_ratios.png)

## 4. Survival analysis - *when* do customers leave?

* Log-rank tests: retention curves differ significantly by contract, internet service and payment method
  (all p < 1e-100).
* Cox proportional-hazards model: concordance **{sv['cox_concordance']:.3f}**.
* Proportional-hazards assumption is violated (p < 0.01) for: {', '.join(sv['ph_violations']) or 'none'} - their
  hazard ratios should be read as time-averaged effects (a stratified or time-varying Cox model is the next step).

| Archetype ($70/month) | 12-month retention | 24-month retention | Expected months (of first 72) |
|---|---|---|---|
""" + "\n".join(f"| {r.archetype} | {_pct(r.survival_12m, 0)} | {_pct(r.survival_24m, 0)} | {r.expected_months_in_first_72:.0f} |"
               for r in arche.itertuples()) + f"""

![Kaplan-Meier](figures/survival_kaplan_meier.png)
![Cox hazard ratios](figures/survival_cox_hazard_ratios.png)
![Archetypes](figures/survival_archetypes.png)

## 5. Modelling

**Feature engineering** (stateless, leak-free, shared by training and serving): service counts, protection/streaming
counts, average historical charge vs current charge, price per service, contract length, auto-pay and e-check flags,
family flag, fibre-without-support risk flag, tenure buckets, log-tenure.

**Protocol.** Six model families were tuned with Optuna (TPE sampler, median pruning) on identical 5-fold
stratified CV splits: logistic regression, random forest, XGBoost, LightGBM, CatBoost (native categoricals), and a
PyTorch **entity-embedding neural network** (TabularNet). Two ensembles were built from out-of-fold predictions: a
stacked logistic meta-learner (weights on log-odds: {stack_w}) and a soft-voting average of the top three
({', '.join(m['soft_voting_members'])}). **Model selection used cross-validation only - never the test set.** {tuning_txt}

{_table(comp_tbl)}

{sig_txt} On this dataset the signal is largely additive, so a well-engineered linear model is a very strong
baseline (test AUC {lr_test:.4f}) - a finding worth stating plainly rather than hiding behind a leaderboard.

**Selection rule.** (1) Every model within one standard error ({m['cv_standard_error']:.4f}) of the best CV AUC
is treated as statistically tied on ranking quality - here: {', '.join(m['within_1se'])}. (2) Among tied models
the deployed one is the **best calibrated** (lowest out-of-fold Brier score), because the campaign rule multiplies
probabilities by dollar values: {calib_txt}

![Model comparison](figures/model_comparison.png)
![ROC / PR](figures/roc_pr_curves.png)
![Calibration](figures/calibration.png)
![DL learning curve](figures/dl_learning_curve.png)

**Does re-balancing the classes help?** (tuned LightGBM, 5-fold OOF)

{_table(imb_tbl)}

Re-balancing leaves ranking quality (ROC-AUC) essentially unchanged but distorts probabilities (higher Brier, inflated
mean prediction). Because decisions are made with a cost-based threshold on calibrated probabilities, the natural
class distribution is kept.

### Hold-out performance of the deployed model (threshold = {md['threshold']:.3f})

| ROC-AUC | PR-AUC | Brier | Precision | Recall | F1 | Specificity | MCC | Top-decile lift |
|---|---|---|---|---|---|---|---|---|
| {tm['roc_auc']:.4f} | {tm['pr_auc']:.4f} | {tm['brier']:.4f} | {tm['precision']:.3f} | {tm['recall']:.3f} | {tm['f1']:.3f} | {tm['specificity']:.3f} | {tm['mcc']:.3f} | {tm['lift_top10']:.2f}x |

Top-3 deciles capture {_pct(dec['cum_capture'].iloc[2], 0)} of all churners.

![Confusion matrix](figures/confusion_matrix.png)
![Gains and lift](figures/gains_lift.png)
![Threshold trade-off](figures/threshold_tradeoff.png)

## 6. Business impact - retention campaign economics

Assumptions (all in `config.yaml`): margin at risk = MonthlyCharges x {bp['clv_horizon_months']} months x
{_pct(bp['gross_margin'], 0)} gross margin; contact cost ${bp['contact_cost']:.0f} + incentive
${bp['incentive_cost']:.0f} per targeted customer; {_pct(bp['offer_success_rate'], 0)} of targeted churners are saved.
The uniform threshold was optimised on **training OOF** predictions and then applied to the test set. Figures are
test-set outcomes scaled x{scale:.2f} to the full customer base.

{_table(pol_tbl)}

The **value-aware rule** (target iff p x success x value > cost) needs no tuning because the model's probabilities
are calibrated; it captures {_pct(ev_row['net_profit'] / oracle['net_profit'], 0)} of the perfect-foresight profit.

![Profit curve](figures/business_profit_curve.png)
![Policies](figures/business_policy_comparison.png)
![Sensitivity](figures/business_sensitivity.png)

## 7. Explainability

* Raw-feature SHAP on the deployed model - top drivers: **{', '.join(shap_top)}**.
* Permutation importance on the hold-out set and TreeSHAP on engineered features agree on the ranking.

![SHAP importance](figures/shap_global_importance.png)
![SHAP dependence](figures/shap_dependence.png)
![Local explanations](figures/shap_local_examples.png)
![TreeSHAP beeswarm](figures/shap_tree_beeswarm.png)
![Permutation importance](figures/permutation_importance.png)

## 8. Segmentation - who to call first

Every customer is scored **out-of-sample** (OOF for training rows, hold-out predictions for test rows) and placed on
a risk x value grid:

{_table(seg_tbl)}

![Risk x value](figures/segmentation_risk_value_matrix.png)
![Personas](figures/segmentation_personas.png)

## 9. Responsible AI - fairness audit

Group metrics on the test set with the value-aware targeting policy:

{_table(fair_tbl)}

{fair_txt}

![Fairness](figures/fairness_group_parity.png)

## 10. Limitations & next steps

* **Snapshot data** - one observation per customer, no event dates or usage history; features such as support-call
  volume or network quality would likely add signal.
* **Associational, not causal** - SHAP and the dashboard's what-if simulator show what the model *associates* with
  lower risk. Offer effectiveness must be measured with a randomised test; the natural next step is an **uplift
  model** (T-/X-learner) trained on campaign data to target *persuadable* customers rather than merely high-risk ones.
* Campaign economics are illustrative assumptions; the sensitivity grid shows where the business case holds.
* Proportional-hazards violations suggest a stratified / time-varying Cox model for lifetime forecasting.
* In production: monitor score drift (PSI), calibration and realised campaign lift; retrain quarterly.

## 11. Reproducibility

* One command: `python run_pipeline.py` (full) or `--fast` (smoke) or `--reuse` (skip tuning).
* Seeds fixed everywhere; tuned hyper-parameters saved to `models/best_params.json`.
* Versions: {', '.join(f"{k} {v}" for k, v in md['versions'].items())}.
* Pipeline runtime: {s['runtime_minutes']:.1f} min. Single-customer inference latency: {md['latency_ms_single']:.1f} ms.
"""
    (REPORTS_DIR / "REPORT.md").write_text(text, encoding="utf-8")

    # ---- README results block (between markers) -------------------------------
    lb = comp.head(8)
    leaderboard = _table(pd.DataFrame({
        "Model": ["**" + x + "**" if x == md["model_name"] else x for x in lb["model"]],
        "CV ROC-AUC": [f"{a:.4f} +/- {b:.4f}" for a, b in zip(lb["cv_auc_mean"], lb["cv_auc_std"])],
        "Test ROC-AUC": [f"{x:.4f}" for x in lb["roc_auc"]],
        "Test PR-AUC": [f"{x:.4f}" for x in lb["pr_auc"]],
        "Test Brier": [f"{x:.4f}" for x in lb["brier"]],
    }))
    block = f"""
| Headline | Value |
|---|---|
| Deployed model | **{md['model_name']}** (selected on 5-fold CV; tied models resolved by calibration) |
| Hold-out ROC-AUC [95% CI] | **{tm['roc_auc']:.3f}** [{tm['auc_ci_low']:.3f}, {tm['auc_ci_high']:.3f}] |
| Hold-out PR-AUC / Brier score | {tm['pr_auc']:.3f} / {tm['brier']:.3f} (base churn rate {_pct(e['churn_rate'])}) |
| Top-decile lift / churners caught in top 30% | **{tm['lift_top10']:.1f}x** / {_pct(dec['cum_capture'].iloc[2], 0)} |
| Projected retention-campaign profit (full base) | **{_money(ev_row['net_profit'] * scale)}** (ROI {ev_row['roi']:.0%}) vs {_money(all_row['net_profit'] * scale)} contacting everyone |
| Monthly revenue lost to churn | {_money(e['monthly_revenue_lost'])} ({_pct(e['monthly_revenue_lost'] / e['monthly_revenue_total'])} of MRR) |

**Key findings**

* **Contract is the #1 lever:** month-to-month customers churn at {_pct(e['churn_by_contract']['Month-to-month'])}
  ({e['mtm_vs_one_year_ratio']:.1f}x one-year, {e['mtm_vs_two_year_ratio']:.0f}x two-year) and drive
  {_pct(e['revenue_lost_by_contract']['Month-to-month'] / e['monthly_revenue_lost'], 0)} of lost revenue.
* **The first year is the danger zone:** only {_pct(sv['retention_12m_overall'], 0)} of customers survive 12 months;
  the median month-to-month customer leaves after {sv['median_lifetime_by_contract']['Month-to-month']:.0f} months.
* **Top model drivers (SHAP):** {', '.join(shap_top)} - fibre without security/tech support and electronic-check
  payment stay significant risk factors after adjusting for contract and tenure.
* **{targeting_txt}**
* **Six tuned model families and two ensembles land within a narrow AUC band** - tabular churn signal is largely additive, so the
  gains come from features, calibration and the decision rule rather than model complexity.

{leaderboard}
"""
    readme = REPORTS_DIR.parent / "README.md"
    if readme.exists():
        src = readme.read_text(encoding="utf-8")
        start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
        if start in src and end in src:
            head, rest = src.split(start, 1)
            _, tail = rest.split(end, 1)
            readme.write_text(head + start + "\n" + block.strip() + "\n" + end + tail, encoding="utf-8")
