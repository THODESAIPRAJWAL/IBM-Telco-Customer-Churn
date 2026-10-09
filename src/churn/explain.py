"""Explainability: global + local, model-agnostic + model-specific.

* **Raw-space SHAP** (permutation explainer) on the *deployed* model - attributions
  in probability points on the original 18 customer attributes.
* **TreeSHAP** on the best gradient-boosting model - exact attributions on the
  engineered feature space, useful to validate the engineered features.
* **Permutation importance** on the hold-out set - how much test AUC drops when a
  raw attribute is shuffled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score

from . import plotting as P
from .config import CATEGORY_LEVELS, MODELS_DIR, TABLES_DIR, load_config
from .predict import MODEL_FEATURES, RawSpaceModel

plt = P.plt


def raw_shap(model, X_background: pd.DataFrame, X_explain: pd.DataFrame, max_evals: int = 200) -> shap.Explanation:
    facade = RawSpaceModel(model)
    bg = facade.encode(X_background[MODEL_FEATURES])
    explainer = shap.PermutationExplainer(facade, shap.maskers.Independent(bg, max_samples=len(bg)), seed=0)
    ex = explainer(facade.encode(X_explain[MODEL_FEATURES]), max_evals=max_evals, silent=True)
    ex.feature_names = MODEL_FEATURES
    return ex


def tree_shap(pipe) -> tuple[shap.TreeExplainer, list[str]]:
    pre = pipe.named_steps["preprocess"]
    names = [n.replace("_", " = ", 1) if n.split("_")[0] in CATEGORY_LEVELS or n.startswith("tenure_bucket")
             else n for n in pre.get_feature_names_out()]
    return shap.TreeExplainer(pipe.named_steps["model"]), names


# ---------------------------------------------------------------------------
def fig_raw_importance(ex: shap.Explanation, model_name: str) -> pd.DataFrame:
    imp = pd.DataFrame({"feature": MODEL_FEATURES, "mean_abs_shap": np.abs(ex.values).mean(axis=0)})
    imp = imp.sort_values("mean_abs_shap", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6.2), layout="constrained")
    ax.barh(imp["feature"], imp["mean_abs_shap"] * 100, color=P.BLUE, height=0.65)
    for i, v in enumerate(imp["mean_abs_shap"] * 100):
        ax.text(v + 0.1, i, f"{v:.1f}", va="center", fontsize=8.5, color=P.INK_2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean |SHAP value| (percentage points of churn probability)")
    top = imp.iloc[::-1].head(3)["feature"].tolist()
    P.title(ax, f"Global feature importance - {model_name}",
            f"Raw-feature SHAP on the deployed model; top drivers: {', '.join(top)}")
    P.savefig(fig, "shap_global_importance")
    return imp.iloc[::-1].reset_index(drop=True)


def fig_raw_dependence(ex: shap.Explanation, X: pd.DataFrame) -> None:
    """How each value of the key attributes moves the churn probability."""
    num = ["tenure", "MonthlyCharges"]
    cats = ["Contract", "InternetService", "PaymentMethod", "OnlineSecurity", "TechSupport", "PaperlessBilling"]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), layout="constrained")
    idx = {f: i for i, f in enumerate(MODEL_FEATURES)}
    for ax, f in zip(axes.flat[:2], num):
        ax.scatter(X[f], ex.values[:, idx[f]] * 100, s=10, alpha=0.5, color=P.BLUE, linewidths=0)
        ax.axhline(0, color=P.INK_2, lw=1)
        ax.set_xlabel(f); ax.set_ylabel("SHAP (pp)")
        ax.set_title(f, fontsize=10.5)
    for ax, f in zip(axes.flat[2:], cats):
        levels = [l for l in CATEGORY_LEVELS[f] if (X[f] == l).any()]
        rng = np.random.default_rng(0)
        for i, lvl in enumerate(levels):
            v = ex.values[(X[f] == lvl).to_numpy(), idx[f]] * 100
            ax.scatter(v, i + rng.uniform(-0.18, 0.18, len(v)), s=8, alpha=0.35,
                       color=P.CHURNED if np.mean(v) > 0 else P.RETAINED, linewidths=0)
            ax.plot(np.mean(v), i, "D", color=P.INK, ms=6)
        ax.axvline(0, color=P.INK_2, lw=1)
        ax.set_yticks(range(len(levels)), [l.replace(" (automatic)", " (auto)") for l in levels], fontsize=8.5)
        ax.set_xlabel("SHAP (pp)")
        ax.grid(axis="y", visible=False)
        ax.set_title(f, fontsize=10.5)
    fig.suptitle("How attribute values push churn probability up (orange) or down (blue); diamond = mean",
                 x=0.01, ha="left", fontsize=13, fontweight="semibold")
    P.savefig(fig, "shap_dependence")


def _waterfall(ax, contrib: pd.DataFrame, base: float, pred: float, title: str, top: int = 9) -> None:
    d = contrib.head(top).iloc[::-1]
    rest = contrib["contribution"].iloc[top:].sum()
    labels = [f"{r.feature} = {r.value}" for r in d.itertuples()]
    vals = d["contribution"].tolist()
    if abs(rest) > 1e-9:
        labels = [f"{len(contrib) - top} other features"] + labels
        vals = [rest] + vals
    cum = base + np.cumsum([0] + vals[:-1])
    # order bottom->top reversed so the biggest driver is on top
    for i, (lab, v, start) in enumerate(zip(labels, vals, cum)):
        c = P.CHURNED if v > 0 else P.RETAINED
        ax.barh(i, v * 100, left=start * 100, color=c, height=0.62)
        ax.text(max(start, start + v) * 100 + 0.8, i, f"{v * 100:+.1f}", va="center", fontsize=8.5, color=P.INK_2)
    ax.set_yticks(range(len(labels)), labels, fontsize=8.5)
    ax.axvline(base * 100, color=P.MUTED, lw=1)
    ax.axvline(pred * 100, color=P.INK, lw=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Churn probability (%)")
    P.title(ax, title, f"baseline {base:.1%} -> prediction {pred:.1%}")


def fig_local_examples(ex: shap.Explanation, X: pd.DataFrame, p: np.ndarray, threshold: float) -> list[dict]:
    order = np.argsort(p)
    picks = {
        "Highest-risk customer": int(order[-1]),
        "Borderline customer": int(np.argmin(np.abs(p - threshold))),
        "Lowest-risk customer": int(order[0]),
    }
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2), layout="constrained")
    out = []
    for ax, (label, i) in zip(axes, picks.items()):
        contrib = pd.DataFrame({"feature": MODEL_FEATURES, "value": X.iloc[i][MODEL_FEATURES].values,
                                "contribution": ex.values[i]})
        contrib = contrib.reindex(contrib["contribution"].abs().sort_values(ascending=False).index)
        base = float(ex.base_values[i])
        _waterfall(ax, contrib, base, base + contrib["contribution"].sum(), label)
        out.append({"case": label, "customerID": X.iloc[i].get("customerID", i), "probability": float(p[i])})
    fig.suptitle("Local explanations: why the model scored these customers the way it did",
                 x=0.01, ha="left", fontsize=13, fontweight="semibold")
    P.savefig(fig, "shap_local_examples")
    return out


def fig_tree_beeswarm(pipe, X: pd.DataFrame, model_name: str) -> pd.DataFrame:
    explainer, names = tree_shap(pipe)
    Z = pipe.named_steps["preprocess"].transform(pipe.named_steps["features"].transform(X))
    sv = explainer(Z)
    sv.feature_names = names
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(sv, max_display=20, show=False, color=P.DIVERGING, plot_size=None)
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_facecolor(P.SURFACE)
    P.title(ax, f"TreeSHAP on engineered features - {model_name}",
            "Each dot is a customer; colour = feature value (blue low, red high); right = raises churn log-odds")
    P.savefig(fig, "shap_tree_beeswarm")
    return (pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv.values).mean(axis=0)})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))


def permutation_importance(model, X: pd.DataFrame, y, n_repeats: int = 10, seed: int = 0) -> pd.DataFrame:
    """Drop in ROC-AUC when one raw attribute is shuffled (works for any predict_proba model)."""
    rng = np.random.default_rng(seed)
    X = X[MODEL_FEATURES].reset_index(drop=True)
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    rows = []
    for f in MODEL_FEATURES:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[f] = rng.permutation(Xp[f].to_numpy())
            drops.append(base - roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
        rows.append({"feature": f, "auc_drop": float(np.mean(drops)), "std": float(np.std(drops))})
    return pd.DataFrame(rows)


def fig_permutation_importance(model, X, y, model_name: str) -> pd.DataFrame:
    d = permutation_importance(model, X, y)
    d = d.sort_values("auc_drop")
    fig, ax = plt.subplots(figsize=(9, 6.2), layout="constrained")
    ax.barh(d["feature"], d["auc_drop"], xerr=d["std"], color=P.BLUE, height=0.65,
            error_kw=dict(ecolor=P.INK_2, lw=1, capsize=0))
    ax.axvline(0, color=P.INK_2, lw=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Drop in test ROC-AUC when the attribute is shuffled (mean +/- sd, 10 repeats)")
    P.title(ax, f"Permutation importance on the hold-out set - {model_name}",
            "Model-agnostic check that agrees with (or challenges) the SHAP ranking")
    P.savefig(fig, "permutation_importance")
    return d.iloc[::-1].reset_index(drop=True)


def run(model, model_name: str, X_train, X_test, y_test, p_test, threshold: float,
        tree_pipe=None, tree_name: str | None = None) -> dict:
    cfg = load_config()["explain"]
    bg = X_train.sample(cfg["shap_background_size"], random_state=0)
    bg[MODEL_FEATURES].to_csv(MODELS_DIR / "shap_background.csv", index=False)
    sample_idx = X_test.sample(min(cfg["shap_sample_size"], len(X_test)), random_state=1).index
    Xs = X_test.loc[sample_idx]
    ex = raw_shap(model, bg, Xs, max_evals=cfg["shap_max_evals"])
    imp = fig_raw_importance(ex, model_name)
    fig_raw_dependence(ex, Xs)
    p_s = pd.Series(p_test, index=X_test.index).loc[sample_idx].to_numpy()
    locals_ = fig_local_examples(ex, Xs, p_s, threshold)
    perm = fig_permutation_importance(model, X_test, y_test, model_name)

    shap_df = pd.DataFrame(ex.values, columns=[f"shap_{f}" for f in MODEL_FEATURES], index=Xs.index)
    pd.concat([Xs[["customerID", *MODEL_FEATURES]], shap_df], axis=1).assign(
        base_value=ex.base_values).to_csv(TABLES_DIR / "shap_values_sample.csv", index=False)
    imp.to_csv(TABLES_DIR / "shap_global_importance.csv", index=False)
    perm.to_csv(TABLES_DIR / "permutation_importance.csv", index=False)

    tree_imp = None
    if tree_pipe is not None:
        tree_imp = fig_tree_beeswarm(tree_pipe, X_test, tree_name)
        tree_imp.to_csv(TABLES_DIR / "shap_tree_importance.csv", index=False)
    return {
        "shap_top_features": imp.head(8).to_dict("records"),
        "permutation_top_features": perm.head(8).to_dict("records"),
        "tree_shap_top_features": tree_imp.head(10).to_dict("records") if tree_imp is not None else None,
        "local_examples": locals_,
    }
