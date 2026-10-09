"""Model evaluation: metrics, uncertainty, significance tests and diagnostic plots."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from . import plotting as P

plt = P.plt


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def lift_at(y: np.ndarray, p: np.ndarray, frac: float = 0.1) -> float:
    """Churn rate among the top `frac` highest-scored customers / base rate."""
    y, p = np.asarray(y), np.asarray(p)
    k = max(1, int(round(frac * len(y))))
    top = np.argsort(-p)[:k]
    return float(y[top].mean() / y.mean())


def metrics_at(y, p, threshold: float = 0.5) -> dict[str, float]:
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)),
        "threshold": threshold,
        "accuracy": (tp + tn) / len(y),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y, pred, zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
        "lift_top10": lift_at(y, p, 0.1),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def bootstrap_ci(y, p, metric=roc_auc_score, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Stratified percentile bootstrap CI for any score(y, p) metric."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    scores = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        scores[b] = metric(y[idx], p[idx])
    return float(np.quantile(scores, alpha / 2)), float(np.quantile(scores, 1 - alpha / 2))


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def delong_test(y, p1, p2) -> dict[str, float]:
    """DeLong et al. (1988) test for two correlated ROC AUCs (fast Sun & Xu algorithm)."""
    y = np.asarray(y)
    order = np.argsort(-y, kind="mergesort")  # positives first
    preds = np.vstack([np.asarray(p1)[order], np.asarray(p2)[order]])
    m = int(y.sum())
    n = len(y) - m
    tx = np.array([_midrank(r[:m]) for r in preds])
    ty = np.array([_midrank(r[m:]) for r in preds])
    tz = np.array([_midrank(r) for r in preds])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    return {"auc_1": float(aucs[0]), "auc_2": float(aucs[1]), "diff": float(aucs[0] - aucs[1]),
            "z": float(z), "p_value": float(2 * stats.norm.sf(abs(z)))}


def best_f1_threshold(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[np.argmax(f1[:-1])])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _emphasis(name: str, highlight: set[str]) -> tuple[str, float, float]:
    if name in highlight:
        return P.MODEL_COLORS.get(name, P.BLUE), 2.2, 1.0
    return P.GHOST, 1.2, 0.9


def fig_model_comparison(cv: pd.DataFrame, test: pd.DataFrame, selected: str) -> None:
    """CV AUC (mean +/- sd across folds) next to hold-out test AUC with 95% CI."""
    cv = cv[cv["key"] != "dummy"]
    order = cv.sort_values("cv_auc_mean")["model"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.5 * len(order) + 1.8), sharey=True, layout="constrained")
    for ax, (df, mean, lo, hi, ttl) in zip(
        axes,
        [
            (cv.set_index("model"), "cv_auc_mean", None, None, "5-fold CV ROC-AUC (training set)"),
            (test.set_index("model"), "roc_auc", "auc_ci_low", "auc_ci_high", "Hold-out test ROC-AUC (95% bootstrap CI)"),
        ],
    ):
        for i, m in enumerate(order):
            c = P.BLUE if m == selected else P.INK_2
            v = df.loc[m, mean]
            if lo is None:
                sd = df.loc[m, "cv_auc_std"]
                a, b = v - sd, v + sd
            else:
                a, b = df.loc[m, lo], df.loc[m, hi]
            ax.plot([a, b], [i, i], color=c if m == selected else P.GHOST, lw=2.5, solid_capstyle="round")
            ax.plot(v, i, "o", color=c, ms=8, markeredgecolor=P.SURFACE, markeredgewidth=2)
            ax.text(b + 0.002, i, f"{v:.4f}", va="center", fontsize=8.5,
                    color=P.INK if m == selected else P.INK_2,
                    fontweight="bold" if m == selected else "normal")
        ax.set_yticks(range(len(order)), order)
        ax.grid(axis="y", visible=False)
        ax.set_title(ttl, fontsize=10.5)
        vals = df.loc[[m for m in order if m in df.index], mean]
        ax.set_xlim(max(0.78, vals[vals > 0.6].min() - 0.02), min(1.0, vals.max() + 0.03))
    fig.suptitle(f"Model comparison - selected model: {selected}", x=0.01, ha="left",
                 fontsize=13, fontweight="semibold")
    P.savefig(fig, "model_comparison")


def fig_roc_pr(y, preds: dict[str, np.ndarray], highlight: set[str]) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.4), layout="constrained")
    items = sorted(preds.items(), key=lambda kv: kv[0] in highlight)  # highlighted drawn last
    for name, p in items:
        c, lw, alpha = _emphasis(name, highlight)
        fpr, tpr, _ = roc_curve(y, p)
        a1.plot(fpr, tpr, color=c, lw=lw, alpha=alpha,
                label=f"{name} ({roc_auc_score(y, p):.3f})" if name in highlight else None)
        prec, rec, _ = precision_recall_curve(y, p)
        a2.plot(rec, prec, color=c, lw=lw, alpha=alpha,
                label=f"{name} ({average_precision_score(y, p):.3f})" if name in highlight else None)
    a1.plot([0, 1], [0, 1], color=P.BASELINE, lw=1)
    a2.axhline(np.mean(y), color=P.BASELINE, lw=1)
    a2.text(0.99, np.mean(y) + 0.015, f"base rate {P.pct(np.mean(y))}", ha="right", fontsize=8.5, color=P.MUTED)
    a1.set_xlabel("False positive rate"); a1.set_ylabel("True positive rate (recall)")
    a2.set_xlabel("Recall"); a2.set_ylabel("Precision")
    a1.legend(loc="lower right", title="ROC-AUC"); a2.legend(loc="upper right", title="PR-AUC")
    P.title(a1, "ROC curves (test set)", "Grey = other candidate models")
    P.title(a2, "Precision-recall curves (test set)", "More informative than ROC under class imbalance")
    P.savefig(fig, "roc_pr_curves")


def fig_calibration(y, preds: dict[str, np.ndarray], highlight: list[str]) -> None:
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 8), height_ratios=[3, 1.1], sharex=True, layout="constrained")
    a1.plot([0, 1], [0, 1], color=P.BASELINE, lw=1, label="Perfectly calibrated")
    for name in highlight:
        p = preds[name]
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        c = P.MODEL_COLORS.get(name, P.BLUE)
        a1.plot(mean, frac, "o-", color=c, ms=6, markeredgecolor=P.SURFACE, markeredgewidth=1.5,
                label=f"{name} (Brier {brier_score_loss(y, p):.4f})")
        a2.hist(p, bins=30, range=(0, 1), color=c, alpha=0.5, edgecolor=P.SURFACE, linewidth=1)
    a1.set_ylabel("Observed churn rate"); a1.legend(loc="upper left")
    a2.set_xlabel("Predicted churn probability"); a2.set_ylabel("Customers")
    a2.grid(axis="x", visible=False)
    P.title(a1, "Reliability diagram (test set, deciles)",
            "Calibrated probabilities are what make the expected-value targeting rule valid")
    P.savefig(fig, "calibration")


def fig_confusion(y, p, threshold: float, name: str) -> None:
    pred = (np.asarray(p) >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.6, 4.8), layout="constrained")
    ax.imshow(cm, cmap=P.SEQUENTIAL, vmin=0, vmax=cm.max() * 1.1)
    labels = [["True negative", "False positive"], ["False negative", "True positive"]]
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            ax.text(j, i, f"{v:,}\n{labels[i][j]}\n{P.pct(v / cm[i].sum())} of row", ha="center",
                    va="center", fontsize=9.5, color="white" if v > cm.max() * 0.55 else P.INK)
    ax.set_xticks([0, 1], ["Predicted: stay", "Predicted: churn"])
    ax.set_yticks([0, 1], ["Actual: stayed", "Actual: churned"])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    P.title(ax, f"Confusion matrix - {name}", f"Profit-optimal threshold = {threshold:.3f} (test set)")
    P.savefig(fig, "confusion_matrix")


def fig_gains_lift(y, p, name: str) -> pd.DataFrame:
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(-p)
    ys = y[order]
    frac = np.arange(1, len(y) + 1) / len(y)
    gains = np.cumsum(ys) / ys.sum()
    deciles = pd.DataFrame({"decile": np.ceil(frac * 10).astype(int), "y": ys}).groupby("decile")["y"].agg(["mean", "sum", "size"])
    deciles["lift"] = deciles["mean"] / y.mean()
    deciles["cum_capture"] = deciles["sum"].cumsum() / ys.sum()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    a1.plot(frac, gains, color=P.BLUE, label=name)
    a1.plot([0, 1], [0, 1], color=P.BASELINE, lw=1, label="Random targeting")
    k = deciles["cum_capture"].iloc[2]
    a1.plot(0.3, k, "o", color=P.BLUE, ms=8, markeredgecolor=P.SURFACE, markeredgewidth=2)
    a1.annotate(f"Top 30% of scores capture {P.pct(k, 0)} of churners", (0.3, k), (0.36, k - 0.14),
                fontsize=9, color=P.INK_2, arrowprops=dict(arrowstyle="-", color=P.MUTED, lw=0.8))
    a1.set_xlabel("Share of customers targeted (highest risk first)"); a1.set_ylabel("Share of churners captured")
    a1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    a1.legend(loc="lower right")
    P.title(a1, "Cumulative gains (test set)")
    a2.bar(deciles.index, deciles["lift"], color=P.BLUE, width=0.62)
    a2.axhline(1, color=P.INK_2, lw=1)
    for d, v in deciles["lift"].items():
        a2.text(d, v + 0.05, f"{v:.1f}x", ha="center", fontsize=8.5, color=P.INK_2)
    a2.set_xticks(deciles.index, [f"D{d}" for d in deciles.index])
    a2.set_xlabel("Risk decile (D1 = highest predicted risk)"); a2.set_ylabel("Lift over base rate")
    a2.grid(axis="x", visible=False)
    P.title(a2, "Decile lift", f"Top decile churns at {deciles['lift'].iloc[0]:.1f}x the average rate")
    P.savefig(fig, "gains_lift")
    return deciles


def fig_threshold_tradeoff(y, p, chosen: float, f1_thr: float) -> None:
    thr = np.linspace(0.02, 0.95, 120)
    rows = [metrics_at(y, p, t) for t in thr]
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    for col, c, lab in [("precision", P.BLUE, "Precision"), ("recall", P.ORANGE, "Recall"), ("f1", P.AQUA, "F1")]:
        ax.plot(thr, d[col], color=c, label=lab)
        ax.text(thr[-1] + 0.01, d[col].iloc[-1], lab, va="center", fontsize=9, color=P.INK_2)
    ax.axvline(chosen, color=P.INK, lw=1)
    ax.text(chosen + 0.01, 0.03, f"profit-optimal {chosen:.2f}", fontsize=8.5, color=P.INK)
    ax.axvline(f1_thr, color=P.MUTED, lw=1)
    ax.text(f1_thr + 0.01, 0.10, f"F1-optimal {f1_thr:.2f}", fontsize=8.5, color=P.MUTED)
    ax.set_xlabel("Decision threshold"); ax.set_ylim(0, 1.02); ax.set_xlim(0, 1.05)
    ax.legend(loc="upper center", ncols=3)
    P.title(ax, "Precision / recall trade-off vs threshold (test set)",
            "The business threshold is chosen from campaign economics, not from F1")
    P.savefig(fig, "threshold_tradeoff")


def fig_learning_curve(history: dict[str, list[float]], best_epoch: int) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    ep = np.arange(1, len(history["train_loss"]) + 1)
    a1.plot(ep, history["train_loss"], color=P.BLUE, label="Train")
    a1.plot(ep, history["val_loss"], color=P.ORANGE, label="Validation")
    a1.axvline(best_epoch, color=P.INK_2, lw=1)
    a1.text(best_epoch + 0.5, max(history["val_loss"]), f"best epoch {best_epoch}", fontsize=8.5, color=P.INK_2, va="top")
    a1.set_xlabel("Epoch"); a1.set_ylabel("Binary cross-entropy"); a1.legend()
    P.title(a1, "TabularNet training curve", "Early stopping on validation loss")
    a2.plot(ep, history["val_auc"], color=P.VIOLET)
    a2.axvline(best_epoch, color=P.INK_2, lw=1)
    a2.set_xlabel("Epoch"); a2.set_ylabel("Validation ROC-AUC")
    P.title(a2, "Validation ROC-AUC by epoch")
    P.savefig(fig, "dl_learning_curve")
