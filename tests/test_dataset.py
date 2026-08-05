"""Tests for regime_lab.dataset (assembly of features + target)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_lab.config import Settings
from regime_lab.dataset import TARGET_COLUMN, assemble_dataset


def _processed_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.DatetimeIndex(pd.bdate_range("2018-01-01", periods=n).values)
    close = np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    return pd.DataFrame(
        {
            "close": close,
            "log_return": np.r_[np.nan, np.diff(np.log(close))],
            "realized_vol_20d": pd.Series(close).pct_change().rolling(20).std().values,
            "regime_label_vol": rng.integers(0, 3, n),
        },
        index=dates,
    ).dropna()


def test_assemble_dataset_has_target_and_b0_columns_without_nans():
    df = assemble_dataset(_processed_frame(), Settings())
    for col in [TARGET_COLUMN, "trailing_ret_5d", "dead_zone_threshold"]:
        assert col in df.columns
    assert not df.isna().any().any()
    assert df[TARGET_COLUMN].dtype.kind == "i"


def test_assemble_dataset_trailing_return_is_trailing():
    """trailing_ret_5d at t must equal the log return over the k days ENDING at t."""
    settings = Settings()
    frame = _processed_frame()
    df = assemble_dataset(frame, settings)

    t = df.index[50]
    pos = frame.index.get_loc(t)
    expected = np.log(frame["close"].iloc[pos]) - np.log(
        frame["close"].iloc[pos - settings.direction_k]
    )
    assert df.loc[t, "trailing_ret_5d"] == float(expected)
