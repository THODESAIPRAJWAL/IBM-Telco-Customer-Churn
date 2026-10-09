"""Retention-campaign economics: from churn probabilities to dollars.

Economics of contacting customer *i* with a retention offer:

* cost ``c`` (contact + incentive) is paid for every targeted customer;
* if *i* would have churned, the offer saves them with probability ``s`` and we
  keep ``v_i`` = MonthlyCharges x horizon x gross margin.

Expected profit of targeting *i* is ``p_i * s * v_i - c``, so with calibrated
probabilities the optimal policy is simply: **target i iff p_i > c / (s * v_i)**.
That value-aware rule is compared against a single tuned threshold, blanket
targeting and a perfect-information oracle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm

from . import plotting as P
from .config import BusinessParams

plt = P.plt


def _money_k(x: float) -> str:
    return f"-${abs(x) / 1000:,.0f}k" if x < 0 else f"${x / 1000:,.0f}k"


def customer_value(X: pd.DataFrame, bp: BusinessParams) -> np.ndarray:
    """Margin at risk if the customer churns."""
    return X["MonthlyCharges"].to_numpy(dtype=float) * bp.clv_horizon_months * bp.gross_margin


def campaign_outcome(y, target_mask, value, bp: BusinessParams) -> dict[str, float]:
    """Realised outcome of a campaign given who was targeted and who actually churned."""
    y, t, v = np.asarray(y), np.asarray(target_mask, bool), np.asarray(value)
    n_t = int(t.sum())
    tp = t & (y == 1)
    expected_saved = bp.offer_success_rate * tp.sum()
    retained_value = float(bp.offer_success_rate * v[tp].sum())
    cost = n_t * bp.cost_per_target
    profit = retained_value - cost
    return {
        "n_targeted": n_t,
        "share_targeted": n_t / len(y),
        "churners_reached": int(tp.sum()),
        "churners_total": int((y == 1).sum()),
        "expected_customers_saved": float(expected_saved),
        "retained_value": retained_value,
        "campaign_cost": float(cost),
        "net_profit": float(profit),
        "roi": float(profit / cost) if cost else 0.0,
    }


def profit_by_threshold(y, p, value, bp: BusinessParams, grid=None) -> pd.DataFrame:
    grid = np.linspace(0.0, 1.0, 201) if grid is None else grid
    rows = []
    for t in grid:
        r = campaign_outcome(y, np.asarray(p) >= t, value, bp)
        r["threshold"] = t
        rows.append(r)
    return pd.DataFrame(rows)


def optimal_threshold(y, p, value, bp: BusinessParams) -> float:
    """Profit-maximising uniform threshold (fit on out-of-fold training predictions)."""
    curve = profit_by_threshold(y, p, value, bp)
    return float(curve.loc[curve["net_profit"].idxmax(), "threshold"])


def ev_targets(p, value, bp: BusinessParams) -> np.ndarray:
    """Value-aware rule: target when expected saved margin exceeds the cost."""
    return np.asarray(p) * bp.offer_success_rate * np.asarray(value) > bp.cost_per_target


def profit_by_depth(y, score, value, bp: BusinessParams) -> pd.DataFrame:
    """Cumulative profit when targeting customers in descending `score` order."""
    y, v = np.asarray(y), np.asarray(value)
    order = np.argsort(-np.asarray(score))
    gain = np.where(y[order] == 1, bp.offer_success_rate * v[order], 0.0) - bp.cost_per_target
    return pd.DataFrame({"share": np.arange(1, len(y) + 1) / len(y), "profit": np.cumsum(gain)})


def compare_policies(y, p, value, threshold: float, bp: BusinessParams) -> pd.DataFrame:
    y = np.asarray(y)
    policies = {
        "No campaign": np.zeros_like(y, bool),
        "Target everyone": np.ones_like(y, bool),
        f"Uniform threshold (p >= {threshold:.2f})": np.asarray(p) >= threshold,
        "Value-aware rule (p*s*v > c)": ev_targets(p, value, bp),
        "Oracle (perfect foresight)": y == 1,
    }
    out = pd.DataFrame({k: campaign_outcome(y, m, value, bp) for k, m in policies.items()}).T
    out.index.name = "policy"
    return out


def sensitivity_grid(y, p, value, bp: BusinessParams,
                     success_rates=(0.15, 0.25, 0.35, 0.45, 0.55),
                     incentive_costs=(20, 40, 60, 90, 120, 160)) -> pd.DataFrame:
    """Profit of the value-aware policy under alternative campaign assumptions."""
    grid = pd.DataFrame(index=list(success_rates), columns=list(incentive_costs), dtype=float)
    for s in success_rates:
        for c in incentive_costs:
            alt = BusinessParams(bp.clv_horizon_months, bp.gross_margin, bp.contact_cost, c, s)
            grid.loc[s, c] = campaign_outcome(y, ev_targets(p, value, alt), value, alt)["net_profit"]
    grid.index.name, grid.columns.name = "offer_success_rate", "incentive_cost"
    return grid


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_profit_curve(y, p, value, bp: BusinessParams, scale: float = 1.0, seed: int = 0) -> None:
    by_p = profit_by_depth(y, p, value, bp)
    by_ev = profit_by_depth(y, np.asarray(p) * np.asarray(value), value, bp)
    rng = np.random.default_rng(seed)
    rand = profit_by_depth(y, rng.random(len(y)), value, bp)
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    ax.plot(rand["share"], rand["profit"] * scale, color=P.GHOST, lw=1.5, label="Random targeting")
    specs = [(by_p, P.BLUE, "Rank by churn probability", (0, -38), "center"),
             (by_ev, P.AQUA, "Rank by expected value (p x value)", (0, 14), "center")]
    for d, c, lab, offset, ha in specs:
        ax.plot(d["share"], d["profit"] * scale, color=c, label=lab)
        i = d["profit"].idxmax()
        ax.plot(d.loc[i, "share"], d.loc[i, "profit"] * scale, "o", color=c, ms=8,
                markeredgecolor=P.SURFACE, markeredgewidth=2, zorder=5)
        ax.annotate(f"max ${d.loc[i, 'profit'] * scale:,.0f} at {P.pct(d.loc[i, 'share'], 0)} contacted",
                    (d.loc[i, "share"], d.loc[i, "profit"] * scale), offset, textcoords="offset points",
                    fontsize=9, color=P.INK_2, ha=ha,
                    bbox=dict(boxstyle="round,pad=0.25", fc=P.SURFACE, ec="none"))
    ax.axhline(0, color=P.INK_2, lw=1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:,.0f}k"))
    ax.set_xlabel("Share of customer base contacted"); ax.set_ylabel("Net campaign profit")
    top = max(by_p["profit"].max(), by_ev["profit"].max()) * scale
    ax.set_ylim(min(0, rand["profit"].min() * scale) - 0.05 * top, top * 1.15)
    ax.legend(loc="lower right")
    sub = "Test-set outcomes scaled to the full 7,043-customer base" if scale != 1 else "Test set"
    P.title(ax, "Profit curve: how deep should the retention campaign go?", sub)
    P.savefig(fig, "business_profit_curve")


def fig_policy_comparison(policies: pd.DataFrame, scale: float = 1.0) -> None:
    d = policies["net_profit"].astype(float) * scale
    fig, ax = plt.subplots(figsize=(10, 3.6), layout="constrained")
    names = d.index[::-1]
    colors = [P.GHOST if ("Oracle" in n or n == "No campaign") else (P.STATUS["critical"] if d[n] < 0 else P.BLUE) for n in names]
    ax.barh(names, d[names], color=colors, height=0.6)
    span = d.abs().max()
    for i, n in enumerate(names):
        v = d[n]
        roi = policies.loc[n, "roi"]
        txt = f"${v:,.0f}" + (f"  (ROI {roi:.0%})" if policies.loc[n, "campaign_cost"] else "")
        ax.text(v + (0.01 * span if v >= 0 else -0.01 * span), i, txt, va="center",
                ha="left" if v >= 0 else "right", fontsize=9, color=P.INK_2)
    ax.axvline(0, color=P.INK_2, lw=1)
    ax.set_xlim(min(0, d.min()) - 0.35 * span, d.max() + 0.35 * span)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:,.0f}k"))
    ax.grid(axis="y", visible=False)
    P.title(ax, "Net profit by targeting policy",
            "Grey = reference points (do nothing / perfect foresight); red = loss-making policy")
    P.savefig(fig, "business_policy_comparison")


def fig_sensitivity(grid: pd.DataFrame, bp: BusinessParams, scale: float = 1.0) -> None:
    g = grid.astype(float) * scale
    lim = np.abs(g.values).max()
    fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
    if g.values.min() >= 0:  # all profitable -> magnitude only: sequential ramp
        cmap, norm = P.SEQUENTIAL, Normalize(vmin=0, vmax=lim)
    else:  # losses present -> diverging around break-even
        cmap, norm = P.DIVERGING.reversed(), TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)
    im = ax.imshow(g.values, cmap=cmap, norm=norm, aspect="auto")
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            v = g.values[i, j]
            ax.text(j, i, _money_k(v), ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > 0.55 * lim else P.INK)
    ax.set_xticks(range(g.shape[1]), [f"${c}" for c in g.columns])
    ax.set_yticks(range(g.shape[0]), [f"{s:.0%}" for s in g.index])
    ax.set_xlabel(f"Incentive cost per targeted customer (+ ${bp.contact_cost:.0f} contact cost)")
    ax.set_ylabel("Offer success rate")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.outline.set_visible(False)
    cb.set_label("Net profit (value-aware policy)")
    cb.formatter = plt.FuncFormatter(lambda x, _: _money_k(x))
    cb.update_ticks()
    worst = g.values.min()
    sub = ("Profitable in every scenario: the value-aware rule simply contacts fewer customers as offers get pricier"
           if worst >= 0 else "Red cells lose money - the business case depends on these assumptions")
    P.title(ax, "Sensitivity: campaign profit under different assumptions", sub)
    P.savefig(fig, "business_sensitivity")
