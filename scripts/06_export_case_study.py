"""Export the case study.

Collects the final tables and figures into ``outputs/case_study/`` (the
one outputs directory that is committed) and renders the reliability
diagram. FINDINGS.md lives beside them.
Requires scripts 03–05 outputs.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from plot_style import MODEL_COLORS, MODEL_PUBLIC_NAMES, apply_style  # noqa: E402

FIGURES = [
    "r1_baselines.png",
    "r2_state_heatmap.png",
    "r3_coverage_abstention.png",
]
TABLES = [
    "metrics_baselines.csv",
    "paired_diffs_baselines.csv",
    "p1_primary.csv",
    "state_skill_diffs.csv",
    "metrics_by_state.csv",
    "n1_fallbacks.csv",
    "conformal_coverage.csv",
    "conformal_state_coverage.csv",
    "conformal_sweep.csv",
]
CLASS_NAMES = {0: "DOWN", 1: "FLAT", 2: "UP"}


def plot_reliability(preds: pd.DataFrame, out_path: Path) -> None:
    """One-vs-rest reliability, B1 vs B2, all pairs/blocks pooled
    (exploratory): the visual argument for scoring probabilities."""
    apply_style()
    bins = np.linspace(0, 1, 11)
    centers = (bins[:-1] + bins[1:]) / 2

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4), sharey=True)
    for ax, (cls, cls_name) in zip(axes, CLASS_NAMES.items(), strict=True):
        col = ["p_down", "p_flat", "p_up"][cls]
        for model in ["B1", "B2"]:
            g = preds[preds["model"] == model]
            p = g[col].to_numpy()
            hit = (g["y_true"] == cls).to_numpy()
            which = np.digitize(p, bins[1:-1])
            xs, ys = [], []
            for b in range(10):
                mask = which == b
                if mask.sum() >= 50:
                    xs.append(centers[b])
                    ys.append(hit[mask].mean())
            ax.plot(xs, ys, color=MODEL_COLORS[model], linewidth=2, marker="o",
                    markersize=4, label=MODEL_PUBLIC_NAMES[model])
        ax.plot([0, 1], [0, 1], color="#52514e", linewidth=0.8, linestyle="--")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"P({cls_name})", fontsize=9)
        ax.set_xlabel("Predicted probability")
    axes[0].set_ylabel("Observed frequency")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Reliability: predicted vs observed (all pairs & blocks pooled, exploratory; "
        "dashed = perfectly calibrated; bins with < 50 days omitted)",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    out_dir = ROOT / "outputs"
    cs_dir = out_dir / "case_study"
    (cs_dir / "figures").mkdir(parents=True, exist_ok=True)
    (cs_dir / "tables").mkdir(parents=True, exist_ok=True)

    missing = [
        f for f in FIGURES + TABLES
        if not ((out_dir / "figures" / f).exists() or (out_dir / f).exists())
    ]
    if missing:
        print(f"Run scripts 03–05 first; missing: {missing}", file=sys.stderr)
        return 1

    preds = pd.read_parquet(out_dir / "predictions_baselines.parquet")
    plot_reliability(preds, out_dir / "figures" / "r4_reliability.png")

    for name in FIGURES + ["r4_reliability.png"]:
        shutil.copy2(out_dir / "figures" / name, cs_dir / "figures" / name)
    for name in TABLES:
        shutil.copy2(out_dir / name, cs_dir / "tables" / name)

    print("Exported to outputs/case_study/:")
    for p in sorted(cs_dir.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(cs_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
