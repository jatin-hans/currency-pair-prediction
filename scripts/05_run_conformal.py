"""Mondrian split conformal selective prediction wrapped around B1.

B1 is the only tier that beat the floor with a CI excluding zero (B2/N1
are overconfident). Conformal is per volatility state; the calibration
window is the last min(250, 30% of train) rows of each block's training
window, with a k-row purge before it. Pre-registered P2: marginal +
per-state coverage at target 80% on Block 6, tolerance ±5pp marginal.
Outputs: outputs/conformal_*.{parquet,csv} + r3_coverage_abstention.png.
Requires scripts 01–02 outputs (data/processed); independent of 03/04.
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

from plot_style import PAIR_COLORS, STATE_NAMES, apply_style  # noqa: E402
from run_common import scope_of  # noqa: E402

from regime_lab.config import Settings  # noqa: E402
from regime_lab.conformal import MondrianConformal  # noqa: E402
from regime_lab.data.splits import split_by_block  # noqa: E402
from regime_lab.dataset import TARGET_COLUMN, load_pair_dataset  # noqa: E402
from regime_lab.eval import block_bootstrap_ci  # noqa: E402
from regime_lab.models import MODEL_FEATURES, _expand_proba, make_b1  # noqa: E402

SWEEP_TARGETS = [round(0.50 + 0.05 * i, 2) for i in range(10)]  # 0.50 … 0.95


def run_conformal(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-day sets at the committed target + the sweep table."""
    day_rows: list[dict[str, object]] = []
    sweep_rows: list[dict[str, object]] = []
    k = settings.direction_k

    for pair in settings.pairs:
        df = load_pair_dataset(pair, settings, ROOT)
        for block in settings.walk_forward_blocks:
            train, test = split_by_block(df, block, purge=k)
            w = min(250, int(0.3 * len(train)))
            calib, proper = train.iloc[-w:], train.iloc[: -(w + k)]

            model = make_b1(settings.seed).fit(
                proper[MODEL_FEATURES], proper[TARGET_COLUMN]
            )
            proba_calib = _expand_proba(model, calib[MODEL_FEATURES])
            proba_test = _expand_proba(model, test[MODEL_FEATURES])
            y_calib = calib[TARGET_COLUMN].to_numpy()
            s_calib = calib["regime_label_vol"].to_numpy()
            y_test = test[TARGET_COLUMN].to_numpy()
            s_test = test["regime_label_vol"].to_numpy()

            for target in sorted({*SWEEP_TARGETS, settings.target_coverage}):
                conf = MondrianConformal(target_coverage=target).calibrate(
                    proba_calib, y_calib, s_calib
                )
                sets = conf.predict_sets(proba_test, s_test)
                answered = sets.sum(axis=1) == 1
                covered = sets[np.arange(len(y_test)), y_test]
                point = proba_test.argmax(axis=1)
                selective_hits = answered & (sets.argmax(axis=1) == y_test)

                if target == settings.target_coverage:
                    for i, date in enumerate(test.index):
                        day_rows.append(
                            {
                                "pair": pair,
                                "block": block.block_id,
                                "date": date,
                                "y_true": int(y_test[i]),
                                "state": int(s_test[i]),
                                "set_down": bool(sets[i, 0]),
                                "set_flat": bool(sets[i, 1]),
                                "set_up": bool(sets[i, 2]),
                                "answered": bool(answered[i]),
                                "covered": bool(covered[i]),
                                "point_pred": int(point[i]),
                            }
                        )
                sweep_rows.append(
                    {
                        "pair": pair,
                        "block": block.block_id,
                        "target": target,
                        "n": len(y_test),
                        "coverage": float(covered.mean()),
                        "abstention": float(1 - answered.mean()),
                        "n_answered": int(answered.sum()),
                        "selective_accuracy": (
                            float(selective_hits.sum() / answered.sum())
                            if answered.any()
                            else np.nan
                        ),
                        "calib_fallback_states": str(conf.fallback_states_),
                    }
                )
        print(f"  {pair}: done")
    return pd.DataFrame(day_rows), pd.DataFrame(sweep_rows)


def _rate_ci(flags: np.ndarray, settings: Settings) -> tuple[float, float]:
    return block_bootstrap_ci(
        lambda idx: float(flags[idx].mean()),
        n=len(flags),
        block_length=settings.bootstrap_block_length,
        n_resamples=settings.bootstrap_n_resamples,
        seed=settings.seed,
    )


def coverage_tables(
    days: pd.DataFrame, settings: Settings
) -> tuple[pd.DataFrame, pd.DataFrame]:
    days = days.assign(scope=days["block"].map(scope_of))
    marginal_rows, state_rows = [], []
    for (pair, scope), g in days.groupby(["pair", "scope"]):
        covered = g["covered"].to_numpy()
        lo, hi = _rate_ci(covered, settings)
        answered = g["answered"].to_numpy()
        sel = g[g["answered"]]
        marginal_rows.append(
            {
                "pair": pair,
                "scope": scope,
                "n": len(g),
                "coverage": covered.mean(),
                "coverage_ci_lo": lo,
                "coverage_ci_hi": hi,
                "abstention": 1 - answered.mean(),
                "n_answered": int(answered.sum()),
                "selective_accuracy": (
                    float((sel["point_pred"] == sel["y_true"]).mean()) if len(sel) else np.nan
                ),
            }
        )
        for state, gs in g.groupby("state"):
            cov = gs["covered"].to_numpy()
            slo, shi = _rate_ci(cov, settings)
            state_rows.append(
                {
                    "pair": pair,
                    "scope": scope,
                    "state": STATE_NAMES[state],
                    "n": len(gs),
                    "coverage": cov.mean(),
                    "coverage_ci_lo": slo,
                    "coverage_ci_hi": shi,
                    "abstention": 1 - gs["answered"].mean(),
                }
            )
    return pd.DataFrame(marginal_rows), pd.DataFrame(state_rows)


def plot_curves(sweep: pd.DataFrame, settings: Settings, out_path: Path) -> None:
    apply_style()
    block6 = sweep[sweep["block"] == 6].groupby(["pair", "target"]).first().reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.9))
    for pair, color in PAIR_COLORS.items():
        g = block6[block6["pair"] == pair].sort_values("target")
        ax1.plot(g["target"], g["coverage"], color=color, linewidth=2, marker="o",
                 markersize=4, label=pair)
        # Selective accuracy over < 20 answered days is noise, not signal —
        # masked rather than plotted (stated in the title).
        ga = g[g["n_answered"] >= 20]
        ax2.plot(ga["abstention"], ga["selective_accuracy"], color=color, linewidth=2,
                 marker="o", markersize=4, label=pair)
        op = ga[ga["target"] == settings.target_coverage]
        ax2.plot(op["abstention"], op["selective_accuracy"], marker="o", markersize=9,
                 color=color, markerfacecolor="white", markeredgewidth=2)

    lo, hi = min(SWEEP_TARGETS), max(SWEEP_TARGETS)
    ax1.plot([lo, hi], [lo, hi], color="#52514e", linewidth=0.8, linestyle="--")
    ax1.axvline(settings.target_coverage, color="#52514e", linewidth=0.8,
                linestyle=":", alpha=0.8)
    ax1.set_xlabel("Target coverage")
    ax1.set_ylabel("Empirical coverage (Block 6)")
    ax1.set_title("Does the wrapper deliver its target?\n(dashed: perfect calibration; "
                  "dotted: 80% operating point)", fontsize=9)
    ax1.legend(fontsize=8)

    ax2.axhline(1 / 3, color="#52514e", linewidth=0.8, linestyle="--", alpha=0.7)
    ax2.set_xlabel("Abstention rate (share of days the model declines to answer)")
    ax2.set_ylabel("Accuracy when answering")
    ax2.set_title("The thesis: abstain more, be right more often when you do answer\n"
                  "(ring: 80% operating point; dashed: chance = 1/3; points with "
                  "< 20 answered days omitted)", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    settings = Settings()
    out_dir = ROOT / "outputs"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Running Mondrian split conformal over B1 …")
    days, sweep = run_conformal(settings)
    days.to_parquet(out_dir / "conformal_days.parquet")
    sweep.to_csv(out_dir / "conformal_sweep.csv", index=False)

    marginal, per_state = coverage_tables(days, settings)
    marginal.to_csv(out_dir / "conformal_coverage.csv", index=False)
    per_state.to_csv(out_dir / "conformal_state_coverage.csv", index=False)

    print(f"\nP2 + marginal coverage at target {settings.target_coverage:.0%}")
    print(marginal.to_string(index=False, float_format=lambda v: f"{v:7.4f}"))
    print("\nPER-STATE coverage (CIs shown — small subsets swing, plan §8)")
    print(per_state.to_string(index=False, float_format=lambda v: f"{v:7.4f}"))

    plot_curves(sweep, settings, fig_dir / "r3_coverage_abstention.png")
    print(f"\nFigure: {fig_dir / 'r3_coverage_abstention.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
