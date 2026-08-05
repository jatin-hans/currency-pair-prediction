"""Condition-aware model N1 + per-state skill analysis.

Runs the pre-registered P1 comparison — paired Brier difference,
condition-aware GBT (N1) vs pooled GBT (B2), per pair, Block 6 as
confirmation — and the exploratory per-state conditional analysis.

Outputs:
  outputs/predictions_n1.parquet     — per-day N1 predictions
  outputs/n1_fallbacks.csv           — which sparsity path fired per pair × block
  outputs/p1_primary.csv             — P1 pre-registered results
  outputs/metrics_by_state.csv       — per-state conditional metrics (exploratory)
  outputs/state_skill_diffs.csv      — per-state B1−B0 Brier diffs (exploratory)
  outputs/figures/r2_state_heatmap.png — state × pair skill heatmap

Requires outputs/predictions_baselines.parquet (run script 03 first).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from plot_style import DIVERGING, STATE_NAMES, apply_style  # noqa: E402
from run_common import paired_diff_table, run_models  # noqa: E402

from regime_lab.config import Settings  # noqa: E402
from regime_lab.eval import paired_brier_diff_ci, summarize_predictions  # noqa: E402


def per_state_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    """Conditional metrics per pair × model × volatility state (exploratory,
    all blocks pooled — small subsets, CIs would be wide; counts printed)."""
    rows = []
    for (pair, model, state), g in preds.groupby(["pair", "model", "state"]):
        proba = g[["p_down", "p_flat", "p_up"]].to_numpy()
        rows.append(
            {"pair": pair, "model": model, "state": STATE_NAMES[state]}
            | summarize_predictions(g["y_true"].to_numpy(), g["y_pred"].to_numpy(), proba)
        )
    return pd.DataFrame(rows)


def per_state_skill_diffs(
    preds: pd.DataFrame, a: str, b: str, settings: Settings
) -> pd.DataFrame:
    """Paired Brier(A)−Brier(B) per pair × state, all blocks pooled.
    Exploratory; the block bootstrap treats the state-subset sequence as
    contiguous, which understates long gaps."""
    rows = []
    for (pair, state), g in preds.groupby(["pair", "state"]):
        wide = g.pivot_table(
            index="date", columns="model", values=["p_down", "p_flat", "p_up", "y_true"]
        ).sort_index()
        y = wide[("y_true", a)].to_numpy().astype(int)
        pa = np.column_stack([wide[(c, a)].to_numpy() for c in ["p_down", "p_flat", "p_up"]])
        pb = np.column_stack([wide[(c, b)].to_numpy() for c in ["p_down", "p_flat", "p_up"]])
        diff, (lo, hi) = paired_brier_diff_ci(
            y, pa, pb,
            block_length=settings.bootstrap_block_length,
            n_resamples=settings.bootstrap_n_resamples,
            seed=settings.seed,
        )
        rows.append(
            {
                "pair": pair,
                "state": STATE_NAMES[state],
                "comparison": f"{a}-{b}",
                "n": len(y),
                "brier_diff": diff,
                "ci_lo": lo,
                "ci_hi": hi,
                "excludes_zero": bool(hi < 0 or lo > 0),
            }
        )
    return pd.DataFrame(rows)


def plot_state_heatmap(diffs: pd.DataFrame, out_path: Path) -> None:
    """State × pair heatmap of B1−B0 paired Brier diff. Diverging palette:
    blue = model beats floor, red = worse than floor, gray = nothing."""
    apply_style()
    pairs = sorted(diffs["pair"].unique())
    states = ["calm", "normal", "turbulent"]
    grid = np.array(
        [
            [
                diffs[(diffs["pair"] == p) & (diffs["state"] == s)]["brier_diff"].iloc[0]
                for p in pairs
            ]
            for s in states
        ]
    )
    cmap = LinearSegmentedColormap.from_list(
        "skill", [DIVERGING["neg"], DIVERGING["mid"], DIVERGING["pos"]]
    )
    vmax = float(np.abs(grid).max())
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    im = ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(pairs)), pairs)
    ax.set_yticks(range(len(states)), states)
    ax.grid(False)
    for i, s in enumerate(states):
        for j, p in enumerate(pairs):
            row = diffs[(diffs["pair"] == p) & (diffs["state"] == s)].iloc[0]
            star = "*" if row["excludes_zero"] else ""
            ax.text(
                j, i, f"{row['brier_diff']:+.3f}{star}\nn={row['n']}",
                ha="center", va="center", fontsize=8, color="#0b0b0b",
            )
    ax.set_title(
        "Where does skill live? Logistic regression vs floor, paired Brier diff\n"
        "(blue = beats the floor; * = 95% CI excludes zero; exploratory, all blocks pooled)",
        fontsize=9,
    )
    fig.colorbar(im, ax=ax, shrink=0.8, label="Brier(B1) − Brier(B0)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    settings = Settings()
    out_dir = ROOT / "outputs"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = out_dir / "predictions_baselines.parquet"
    if not baseline_path.exists():
        print("Run scripts/03_run_baselines.py first.", file=sys.stderr)
        return 1
    baseline_preds = pd.read_parquet(baseline_path)

    print("Running condition-aware model N1 …")
    n1_preds, info = run_models(settings, ["N1"], ROOT)
    n1_preds.to_parquet(out_dir / "predictions_n1.parquet")
    info.to_csv(out_dir / "n1_fallbacks.csv", index=False)
    print("\nSPARSITY RULE — states that fell back to the pooled model, per fit:")
    print(info.to_string(index=False))

    both = pd.concat([baseline_preds, n1_preds], ignore_index=True)

    # P1 — pre-registered primary comparison (Block 6 = confirmation).
    p1 = paired_diff_table(both, [("N1", "B2")], settings)
    p1.insert(0, "registered", np.where(p1["scope"] == "block6_confirmation", "P1", "exploratory"))
    p1.to_csv(out_dir / "p1_primary.csv", index=False)
    print("\nP1: Brier(N1) − Brier(B2), negative = conditioning helped")
    print(p1.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    # Exploratory conditional analysis.
    state_metrics = per_state_metrics(both)
    state_metrics.to_csv(out_dir / "metrics_by_state.csv", index=False)

    diffs = per_state_skill_diffs(baseline_preds, "B1", "B0", settings)
    diffs.to_csv(out_dir / "state_skill_diffs.csv", index=False)
    print("\nPER-STATE SKILL (B1−B0 paired Brier diff; exploratory)")
    print(diffs.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    plot_state_heatmap(diffs, fig_dir / "r2_state_heatmap.png")
    print(f"\nFigure: {fig_dir / 'r2_state_heatmap.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
