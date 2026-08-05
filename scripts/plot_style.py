"""Shared plotting style for all result figures.

Categorical slots in fixed order assigned to the methods ladder (color
follows the entity — a model keeps its color in every figure), sequential
blue ramp for magnitude
heatmaps, diverging blue↔red with a neutral gray midpoint for
better/worse-than-baseline polarity.
"""

from __future__ import annotations

import matplotlib as mpl

# Categorical slots 1–5 (light mode), one per ladder tier, never re-ordered.
MODEL_COLORS = {
    "B0": "#2a78d6",
    "B1": "#eb6834",
    "B2": "#1baf7a",
    "N1": "#eda100",
    "N3": "#e87ba4",
}
MODEL_PUBLIC_NAMES = {
    "B0": "Majority + persistence",
    "B1": "Logistic regression",
    "B2": "Gradient boosting",
    "N1": "Condition-aware GBT",
    "N3": "Conformal wrapper",
}

# Sequential blue ramp, steps 100→700 (magnitude heatmaps).
SEQ_BLUES = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]

# Diverging pair (polarity): blue = better than comparator, red = worse,
# neutral gray midpoint.
DIVERGING = {"neg": "#2a78d6", "mid": "#f0efec", "pos": "#e34948"}

# Pair-entity colors (fixed, used wherever the series is a currency pair).
PAIR_COLORS = {"EURUSD": "#2a78d6", "USDJPY": "#eb6834", "USDINR": "#1baf7a"}

STATE_NAMES = {0: "calm", 1: "normal", 2: "turbulent"}
DIRECTION_NAMES = {0: "DOWN", 1: "FLAT", 2: "UP"}

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": TEXT_SECONDARY,
            "axes.labelcolor": TEXT_PRIMARY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.linewidth": 0.3,
            "grid.alpha": 0.35,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "text.color": TEXT_PRIMARY,
            "legend.frameon": False,
        }
    )
