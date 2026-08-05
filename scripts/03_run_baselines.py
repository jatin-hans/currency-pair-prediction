"""Run the baseline ladder B0/B1/B2 on all pairs × blocks.

Outputs:
  outputs/predictions_baselines.parquet — per-day predictions, long format
  outputs/metrics_baselines.csv         — pair × model × scope metrics
  outputs/paired_diffs_baselines.csv    — paired Brier diffs, bootstrap CIs
  outputs/figures/r1_baselines.png      — metric bars, both scopes
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from plot_style import MODEL_COLORS, MODEL_PUBLIC_NAMES, apply_style  # noqa: E402
from run_common import metrics_table, paired_diff_table, run_models  # noqa: E402

from regime_lab.config import Settings  # noqa: E402

BASELINES = ["B0", "B1", "B2"]


def plot_metric_bars(metrics: pd.DataFrame, out_path: Path) -> None:
    apply_style()
    scopes = ["blocks1-5_exploratory", "block6_confirmation"]
    scope_titles = {
        "blocks1-5_exploratory": "Blocks 1–5 (exploratory)",
        "block6_confirmation": "Block 6 (confirmation set)",
    }
    metric_specs = [
        ("balanced_accuracy", "Balanced accuracy (dashed: chance = 1/3)", 1 / 3),
        ("brier", "Multiclass Brier, lower is better (dashed: uniform = 2/3)", 2 / 3),
    ]
    models = [m for m in MODEL_COLORS if m in set(metrics["model"])]
    pairs = sorted(metrics["pair"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.2))
    for col, scope in enumerate(scopes):
        for row_i, (metric, title, refline) in enumerate(metric_specs):
            ax = axes[row_i][col]
            sub = metrics[metrics["scope"] == scope]
            width = 0.8 / len(models)
            x = np.arange(len(pairs))
            for j, model in enumerate(models):
                vals = [
                    sub[(sub["pair"] == p) & (sub["model"] == model)][metric].iloc[0]
                    for p in pairs
                ]
                bars = ax.bar(
                    x + j * width,
                    vals,
                    width * 0.9,
                    color=MODEL_COLORS[model],
                    label=MODEL_PUBLIC_NAMES[model] if (row_i == 0 and col == 0) else None,
                )
                ax.bar_label(bars, fmt="%.2f", fontsize=6.5, color="#52514e", padding=1)
            ax.axhline(refline, color="#52514e", linewidth=0.8, linestyle="--", alpha=0.7)
            ax.set_xticks(x + 0.4 - width / 2)
            ax.set_xticklabels(pairs)
            ax.set_title(f"{title} — {scope_titles[scope]}", fontsize=9)
            ax.grid(axis="x", alpha=0)
    fig.legend(loc="lower center", ncol=len(models), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    settings = Settings()
    out_dir = ROOT / "outputs"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Running baselines B0/B1/B2 …")
    preds, _ = run_models(settings, BASELINES, ROOT)
    preds.to_parquet(out_dir / "predictions_baselines.parquet")

    metrics = metrics_table(preds)
    metrics.to_csv(out_dir / "metrics_baselines.csv", index=False)
    print("\nMETRICS (per pair × model × scope)")
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:7.4f}"))

    diffs = paired_diff_table(
        preds, [("B1", "B0"), ("B2", "B0"), ("B2", "B1")], settings
    )
    diffs.to_csv(out_dir / "paired_diffs_baselines.csv", index=False)
    print("\nPAIRED BRIER DIFFERENCES (negative = first model better; exploratory)")
    print(diffs.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    plot_metric_bars(metrics, fig_dir / "r1_baselines.png")
    print(f"\nFigure: {fig_dir / 'r1_baselines.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
