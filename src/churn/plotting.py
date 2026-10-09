"""One visual system for every figure in the project.

Palette: a validated, colour-vision-deficiency-safe categorical order, one-hue
sequential ramp and a blue<->red diverging pair with a neutral grey midpoint.
Colour follows the *entity* (retained is always blue, churned always orange).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .config import FIGURES_DIR  # noqa: E402

# Categorical slots (fixed order - never cycled)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#4a3aa7", "#e34948"
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

# Chrome & ink
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
NEUTRAL_MID = "#f0efec"
GHOST = "#c9c8c1"  # de-emphasised series

# Status (reserved for risk tiers - always paired with a label)
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

RETAINED, CHURNED = BLUE, ORANGE
CHURN_COLORS = {0: RETAINED, 1: CHURNED, "No": RETAINED, "Yes": CHURNED}

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
DIVERGING = LinearSegmentedColormap.from_list("div_blue_red", [BLUE, NEUTRAL_MID, RED])

# Model identity colours (stable across every chart)
MODEL_COLORS = {
    "Logistic Regression": BLUE,
    "Stacked Ensemble": ORANGE,
    "Random Forest": AQUA,
    "XGBoost": YELLOW,
    "LightGBM": MAGENTA,
    "CatBoost": GREEN,
    "TabularNet (DL)": VIOLET,
    "Soft-Voting Ensemble": RED,
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "patch.linewidth": 0,
        }
    )


def title(ax, text: str, sub: str | None = None) -> None:
    """Title plus an optional secondary line that carries the takeaway."""
    ax.set_title(text, pad=26 if sub else 12)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9, color=INK_2, va="bottom")


def savefig(fig, name: str, directory: Path = FIGURES_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


apply_style()
