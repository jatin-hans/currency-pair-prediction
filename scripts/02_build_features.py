"""Build the feature + label parquet per pair (spec §4.3, §5).

Reads ``data/raw/``, writes ``data/processed/{pair}.parquet``, and emits
diagnostic figures alongside the processed data.
Also prints a per-pair summary table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime_lab.config import Settings  # noqa: E402
from regime_lab.data.features import build_features  # noqa: E402
from regime_lab.data.labels import label_regime_vol  # noqa: E402
from regime_lab.data.loaders import _sanitize_ticker  # noqa: E402

REGIME_PALETTE = {0: "#8ecae6", 1: "#ffb703", 2: "#d00000"}
REGIME_NAMES = {0: "low-vol", 1: "medium-vol", 2: "high-vol"}


def _load_raw(settings: Settings, name: str) -> pd.DataFrame:
    raw_dir = ROOT / settings.data_root / "raw"
    ticker = {**settings.pairs, **settings.auxiliary_tickers}[name]
    path = raw_dir / f"{_sanitize_ticker(ticker)}.parquet"
    return pd.read_parquet(path)


def _plot_pseudo_label(
    pair: str, processed: pd.DataFrame, out_path: Path
) -> None:
    vol = processed["realized_vol_20d"]
    labels = processed["regime_label_vol"]

    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    ax.plot(vol.index, vol.values, color="#023047", linewidth=0.9, label="realized_vol_20d")

    # Shade background by regime.
    for regime, color in REGIME_PALETTE.items():
        mask = labels == regime
        if not mask.any():
            continue
        ax.fill_between(
            labels.index,
            0,
            vol.max() * 1.05,
            where=mask,
            color=color,
            alpha=0.18,
            step="post",
            linewidth=0,
            label=REGIME_NAMES[regime],
        )

    ax.set_title(f"{pair}: realized_vol_20d and regime_label_vol (spec §5.1)")
    ax.set_ylabel("realized_vol_20d (stdev of daily log-return)")
    ax.set_ylim(0, vol.max() * 1.05)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.legend(loc="upper left", ncol=4, fontsize=8, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_walk_forward(
    pair: str, processed: pd.DataFrame, settings: Settings, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=150)

    for block in settings.walk_forward_blocks:
        y = block.block_id
        train_start = max(pd.Timestamp(block.train_start), processed.index.min())
        train_end = min(pd.Timestamp(block.train_end), processed.index.max())
        test_start = max(pd.Timestamp(block.test_start), processed.index.min())
        test_end = min(pd.Timestamp(block.test_end), processed.index.max())
        if train_end > train_start:
            ax.barh(
                y,
                (train_end - train_start).days,
                left=train_start,
                height=0.5,
                color="#219ebc",
                label="train" if block.block_id == 1 else None,
            )
        if test_end > test_start:
            ax.barh(
                y,
                (test_end - test_start).days,
                left=test_start,
                height=0.5,
                color="#fb8500",
                label="test" if block.block_id == 1 else None,
            )

    ax.set_yticks([b.block_id for b in settings.walk_forward_blocks])
    ax.set_yticklabels([f"block {b.block_id}" for b in settings.walk_forward_blocks])
    ax.invert_yaxis()
    ax.set_title(f"{pair}: walk-forward blocks (spec §4.4)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, axis="x", linewidth=0.3, alpha=0.4)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    settings = Settings()
    processed_dir = ROOT / settings.data_root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = ROOT / "docs" / "approvals" / "phase-1-artifacts"
    figures_dir.mkdir(parents=True, exist_ok=True)

    dxy = _load_raw(settings, "DXY")
    vix = _load_raw(settings, "VIX")

    summary_rows: list[dict[str, object]] = []

    for pair in settings.pairs:
        raw_pair = _load_raw(settings, pair)
        features = build_features(raw_pair, dxy, vix)

        # Compute labels on the FULL-range realized-vol series (only a
        # 20-day rolling warmup) rather than the post-feature-dropna vol
        # column (which additionally waits for the 252-day return_z_252d
        # warmup to clear). This saves ~1 year of labelled history and
        # keeps block 1's training window usable.
        common_idx = raw_pair.index.intersection(dxy.index).intersection(vix.index)
        pair_aligned = raw_pair.loc[common_idx].sort_index()
        log_ret = np.log(pair_aligned["close"] / pair_aligned["close"].shift(1))
        full_vol = log_ret.rolling(20).std().dropna()
        full_vol.name = "realized_vol_20d"

        label_vol = label_regime_vol(
            full_vol,
            min_window=settings.tercile_min_window,
            smoothing_window=settings.label_smoothing_window,
        )
        processed = features.join(label_vol, how="inner")

        out_path = processed_dir / f"{pair}.parquet"
        processed.to_parquet(out_path)

        # Diagnostic figures.
        _plot_pseudo_label(
            pair, processed, figures_dir / f"pseudo_label_{pair}.png"
        )
        _plot_walk_forward(
            pair, processed, settings, figures_dir / f"walk_forward_{pair}.png"
        )

        class_counts = processed["regime_label_vol"].value_counts().to_dict()
        total = int(len(processed))
        summary_rows.append(
            {
                "pair": pair,
                "rows": total,
                "date_start": processed.index.min().date().isoformat(),
                "date_end": processed.index.max().date().isoformat(),
                "NaN_any": int(processed.isna().any(axis=1).sum()),
                "class_0_pct": 100 * class_counts.get(0, 0) / total,
                "class_1_pct": 100 * class_counts.get(1, 0) / total,
                "class_2_pct": 100 * class_counts.get(2, 0) / total,
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    print("=" * 78)
    print("PER-PAIR SUMMARY (approval report §3.1)")
    print("=" * 78)
    print(summary_df.to_string(index=False,
          float_format=lambda x: f"{x:6.2f}"))
    print()
    print("Processed parquets:")
    for path in sorted(processed_dir.glob("*.parquet")):
        print(f"  {path.name:<16} {path.stat().st_size/1024:>8.1f} KB")
    print()
    print("Figures:")
    for path in sorted(figures_dir.glob("*.png")):
        print(f"  {path.relative_to(ROOT)}")

    # Persist the tables so the report can embed them verbatim.
    summary_df.to_csv(figures_dir / "summary_table.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
