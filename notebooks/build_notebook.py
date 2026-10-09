"""Build (and optionally execute) the narrative walkthrough notebook.

    python notebooks/build_notebook.py            # write + execute notebooks/telco_churn_analysis.ipynb

The notebook reads the artefacts produced by `run_pipeline.py`; every number it shows is computed, so the
narrative never drifts from the results.
"""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# Telco Customer Churn - end-to-end analysis

**Question:** which customers are about to leave, why, and is it worth trying to keep them?

This notebook walks through the full project on the IBM Telco Customer Churn dataset (7,043 customers):

1. Data quality and cleaning
2. Statistical exploratory analysis (hypothesis tests with effect sizes)
3. Survival analysis - *when* customers leave
4. Modelling - six Optuna-tuned model families incl. a PyTorch entity-embedding network, plus two ensembles
5. Evaluation - uncertainty, significance tests, calibration
6. Business economics - turning probabilities into a profitable retention campaign
7. Explainability - global and per-customer SHAP
8. Segmentation and fairness
9. Live scoring with next-best-action simulation

The heavy lifting lives in the tested `src/churn` package and `run_pipeline.py`; this notebook loads the
artefacts it produced and tells the story.
""")

code("""
import json, sys
from pathlib import Path
import pandas as pd
from IPython.display import Image, display, Markdown

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from churn import data
from churn.predict import ChurnPredictor

FIG, TAB = ROOT / "reports" / "figures", ROOT / "reports" / "tables"
M = json.loads((ROOT / "reports" / "metrics.json").read_text())
show = lambda name, width=1000: display(Image(filename=str(FIG / f"{name}.png"), width=width))
pd.set_option("display.precision", 4)
pd.set_option("display.max_columns", 30)
""")

md("## 1. Data quality")
code("""
raw = data.load_raw()
print(raw.shape)
data.data_quality_report(raw)
""")
md("""
`TotalCharges` is stored as text and has blank strings. All of them belong to customers with `tenure == 0` -
they simply haven't been billed yet - so the right fix is **0**, not mean/median imputation.
""")
code("""
print(raw.loc[raw["TotalCharges"].str.strip() == "", ["customerID", "tenure", "MonthlyCharges", "TotalCharges"]].head())
df = data.load_clean()
print("churn rate:", round(df["Churn"].mean(), 4), "| duplicate IDs:", df["customerID"].duplicated().sum())
""")

md("""
## 2. Statistical EDA

Each categorical attribute gets a chi-square test of independence (Bonferroni-corrected) and a
bias-corrected **Cramer's V** effect size; numeric attributes get Mann-Whitney U with a rank-biserial effect size.
""")
code("""pd.read_csv(TAB / "stats_categorical_tests.csv")[["feature", "cramers_v", "p_bonferroni", "highest_churn_level", "highest_churn_rate", "significant"]]""")
code('show("eda_churn_rate_by_category", 1200)')
code('pd.read_csv(TAB / "stats_numeric_tests.csv")')
code('show("eda_numeric_distributions", 1200); show("eda_churn_by_tenure_and_contract")')
code("""
e = M["eda"]
display(Markdown(f"Month-to-month customers churn **{e['mtm_vs_one_year_ratio']:.1f}x** more than one-year and "
                 f"**{e['mtm_vs_two_year_ratio']:.0f}x** more than two-year customers; churners took "
                 f"**${e['monthly_revenue_lost']:,.0f}** of monthly recurring revenue with them."))
show("eda_revenue_at_risk", 850); show("eda_protection_services", 800)
""")
md("""
Marginal churn rates confound factors (fibre customers also tend to be on month-to-month contracts). An
**inferential logistic regression** on a curated, non-collinear design isolates each factor's adjusted effect:
""")
code('display(pd.read_csv(TAB / "stats_logit_odds_ratios.csv")); show("eda_odds_ratios", 850)')
code('show("eda_association_matrix", 850); show("eda_mutual_information", 800)')

md("""
## 3. Survival analysis

Treating `tenure` as time-to-event (active customers are right-censored) answers *when* customers leave.
""")
code("""
s = M["survival"]
print("12-month retention:", round(s["retention_12m_overall"], 3), "| Cox concordance:", round(s["cox_concordance"], 3))
print("Median lifetime by contract (months; None = never reaches 50%):", s["median_lifetime_by_contract"])
print("PH-assumption violations (p<0.01):", s["ph_violations"])
show("survival_kaplan_meier", 1200); show("survival_cox_hazard_ratios", 800); show("survival_archetypes", 850)
""")

md("""
## 4. Modelling

* **Features:** stateless, leak-free domain features (service counts, price change vs. historical average, price per
  service, commitment, auto-pay, fibre-without-support flag, tenure buckets).
* **Models:** logistic regression, random forest, XGBoost, LightGBM, CatBoost and **TabularNet** - a PyTorch MLP
  with entity embeddings for categoricals - each tuned with Optuna (TPE + median pruning) on identical 5-fold
  stratified CV splits; plus a stacked (logistic meta-learner on out-of-fold log-odds) and a soft-voting ensemble.
* **Selection on cross-validation only** - best CV AUC, ties within one standard error broken by calibration; the
  20% hold-out set is scored once at the end.
""")
code("""
cv = pd.DataFrame(M["modelling"]["cv"])[["model", "cv_auc_mean", "cv_auc_std", "oof_pr_auc", "oof_brier"]]
test = pd.DataFrame(M["modelling"]["test"])[["model", "roc_auc", "auc_ci_low", "auc_ci_high", "pr_auc", "brier"]]
cv.merge(test, on="model").sort_values("cv_auc_mean", ascending=False)
""")
code('show("model_comparison", 1150)')
md("Are the differences real? DeLong's test for correlated ROC curves (deployed model vs. each alternative):")
code('pd.DataFrame(M["modelling"]["delong"])')
md("""
On this dataset the signal is largely additive, so the models cluster tightly and a well-built logistic regression
is a very strong baseline. What separates them in practice is **calibration** (Brier score) - and calibrated
probabilities are exactly what the value-aware targeting rule in section 6 needs.
""")
code('show("roc_pr_curves", 1150); show("calibration", 650)')
code('show("dl_learning_curve", 1000)')
md("**Class imbalance:** does re-weighting or SMOTE help?")
code('pd.DataFrame(M["modelling"]["imbalance"])')
md("""
Re-balancing doesn't improve ranking (AUC) but inflates probabilities and worsens the Brier score. Imbalance is a
*decision-threshold* problem, so the natural distribution is kept and the threshold comes from the economics.
""")

md("## 5. Hold-out evaluation of the deployed model")
code("""
md_ = M["metadata"]
print(md_["model_name"], "| profit-optimal threshold:", round(md_["threshold"], 3), "| F1-optimal:", round(md_["f1_threshold"], 3))
pd.Series(md_["test_metrics"]).round(4)
""")
code('show("confusion_matrix", 520); show("gains_lift", 1150); show("threshold_tradeoff", 950)')

md("""
## 6. Business economics

Margin at risk = MonthlyCharges x horizon x gross margin. Targeting costs contact + incentive; a share of targeted
churners is saved. With calibrated probabilities the profit-maximising rule needs no tuning:
**contact customer *i* iff p_i x success x value_i > cost**.
""")
code("""
print(md_["business_params"])
pol = pd.DataFrame(M["business"]["policies_test"]).set_index("policy")
scale = M["business"]["scale_to_full_base"]
(pol[["n_targeted", "churners_reached", "campaign_cost", "retained_value", "net_profit"]] * scale).round(0).join(pol["roi"].round(2))
""")
code('show("business_profit_curve", 1000); show("business_policy_comparison", 1000); show("business_sensitivity", 900)')

md("""
## 7. Explainability

SHAP values are computed on the **deployed** model in the space of the original attributes (categoricals exposed to
the explainer as codes and decoded before prediction), so contributions read as percentage points of churn
probability. Permutation importance on the hold-out set and TreeSHAP on the best GBM are cross-checks.
""")
code('show("shap_global_importance", 850); show("shap_dependence", 1250)')
code('show("shap_local_examples", 1250); show("permutation_importance", 850); show("shap_tree_beeswarm", 850)')

md("## 8. Segmentation and fairness")
code("""
display(pd.read_csv(TAB / "segment_summary.csv"))
show("segmentation_risk_value_matrix", 1000); show("segmentation_personas", 1200)
""")
code("""
display(pd.read_csv(TAB / "fairness_disparity.csv"))
print("gender ablation:", M["fairness"]["gender_ablation"])
show("fairness_group_parity", 1200)
""")

md("""
## 9. Live scoring

The deployed artefact is the full inference path (feature engineering included), wrapped by `ChurnPredictor`.
""")
code("""
pred = ChurnPredictor.load(ROOT / "models")
_, X_test, _, y_test = data.split(df)
pred.score(X_test.head(8)).assign(actual_churn=y_test.head(8).values)
""")
code("""
risky = X_test.iloc[[int(pred.predict_proba(X_test).argmax())]]  # the highest-risk hold-out customer
display(risky.T)
display(pred.explain(risky).head(8))
pred.next_best_actions(risky)
""")
md("""
The next-best-action table is a **model-implied** simulation - it ranks which offers are worth testing. Proving
causal lift needs a randomised campaign and, ultimately, an uplift model trained on its results.

## Conclusions

* Churn is concentrated in **new, month-to-month, fibre customers without protection add-ons who pay by electronic
  check** - consistent across descriptive stats, adjusted odds ratios, Cox hazard ratios and SHAP.
* Six tuned model families and two ensembles land within a narrow AUC band; the deployed model was chosen by CV, and its edge over
  logistic regression is tested, not assumed.
* The value of the model is in the **decision**: value-aware targeting on calibrated probabilities beats blanket
  campaigns and uniform thresholds, and stays profitable across a wide range of offer costs and success rates.
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = HERE / "telco_churn_analysis.ipynb"
nbf.write(nb, out)

if __name__ == "__main__":
    from nbconvert.preprocessors import ExecutePreprocessor

    ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(HERE)}})
    nbf.write(nb, out)
    print("wrote and executed", out)
