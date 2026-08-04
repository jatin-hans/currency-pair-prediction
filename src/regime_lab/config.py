"""Central configuration for Regime Lab.

One source of truth for every default the pipeline uses.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class WalkForwardBlock(BaseModel):
    """One walk-forward train/test block (spec §4.4)."""

    model_config = ConfigDict(frozen=True)

    block_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date


def _default_walk_forward_blocks(date_start: date) -> list[WalkForwardBlock]:
    """Six non-overlapping blocks copied verbatim from spec §4.4."""
    spans: list[tuple[date, date, date]] = [
        (date(2017, 6, 30), date(2017, 7, 1), date(2018, 6, 30)),
        (date(2018, 6, 30), date(2018, 7, 1), date(2019, 12, 31)),
        (date(2019, 12, 31), date(2020, 1, 1), date(2021, 6, 30)),
        (date(2021, 6, 30), date(2021, 7, 1), date(2023, 6, 30)),
        (date(2023, 6, 30), date(2023, 7, 1), date(2024, 12, 31)),
        (date(2024, 12, 31), date(2025, 1, 1), date(2026, 4, 18)),
    ]
    return [
        WalkForwardBlock(
            block_id=i + 1,
            train_start=date_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )
        for i, (train_end, test_start, test_end) in enumerate(spans)
    ]


class Settings(BaseModel):
    """Top-level project configuration.

    Instantiate with no arguments to get the spec defaults; pass keyword
    arguments to override for tests or experiments.
    """

    model_config = ConfigDict(frozen=True)

    # Paths (spec §10.3: keep them relative so the repo is portable).
    data_root: Path = Path("data")
    output_root: Path = Path("outputs")

    # Pairs and tickers (spec §4.1).
    pairs: dict[str, str] = Field(
        default_factory=lambda: {
            "EURUSD": "EURUSD=X",
            "USDINR": "INR=X",
            "USDJPY": "JPY=X",
        }
    )
    auxiliary_tickers: dict[str, str] = Field(
        default_factory=lambda: {
            "DXY": "DX-Y.NYB",
            "VIX": "^VIX",
        }
    )

    # Date range. date_end feeds the data download and the live dashboard;
    # the frozen evaluation blocks below are pinned independently, so
    # extending it does not move Block 6.
    date_start: date = date(2015, 1, 1)
    date_end: date = date(2026, 8, 1)

    # Volatility-state label (plan §5).
    tercile_min_window: int = 252
    label_smoothing_window: int = 5

    # Direction task (plan §7).
    direction_k: int = 5
    dead_zone_mult: float = 0.25
    vol_window: int = 20

    # Conformal + bootstrap (plan §4, §7, §8).
    target_coverage: float = 0.80
    bootstrap_block_length: int = 10
    bootstrap_n_resamples: int = 1000

    # Walk-forward blocks (plan §7 / spec §4.4).
    walk_forward_blocks: list[WalkForwardBlock] = Field(
        default_factory=lambda: _default_walk_forward_blocks(date(2015, 1, 1))
    )

    # Reproducibility.
    seed: int = 42
