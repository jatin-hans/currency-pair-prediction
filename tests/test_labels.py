"""Tests for regime_lab.data.labels.

Volatility-state label is plan §5 (expanding tercile + trailing median
smoothing — trailing is the R0 §6.1 causality fix, since the label is now a
decision-time feature). Direction label is plan §7 (k-day-ahead UP/FLAT/DOWN
with a trailing-vol dead zone).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lab.data.labels import label_direction, label_regime_vol


def _monotone_vol_series(n: int = 1000) -> pd.Series:
    dates = pd.DatetimeIndex(pd.bdate_range("2015-01-01", periods=n).values)
    return pd.Series(np.linspace(0.001, 0.05, n), index=dates, name="realized_vol_20d")


def _random_vol_series(n: int = 1000, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.DatetimeIndex(pd.bdate_range("2015-01-01", periods=n).values)
    return pd.Series(np.abs(rng.normal(0.01, 0.003, n)), index=dates)


def _random_walk_close(n: int = 1000, seed: int = 0, drift: float = 0.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.DatetimeIndex(pd.bdate_range("2015-01-01", periods=n).values)
    log_price = np.cumsum(rng.normal(drift, 0.005, n))
    return pd.Series(np.exp(log_price), index=dates, name="close")


# ---------------------------------------------------------------- vol label


def test_label_regime_vol_returns_integer_series():
    vol = _random_vol_series()
    labels = label_regime_vol(vol, min_window=252, smoothing_window=5)
    assert labels.dtype.kind == "i"


def test_label_regime_vol_values_all_in_zero_one_two():
    vol = _random_vol_series()
    labels = label_regime_vol(vol, min_window=252, smoothing_window=5)
    assert set(labels.unique()).issubset({0, 1, 2})


def test_label_regime_vol_drops_rows_before_min_window():
    """Rows with <min_window observations have no tercile — they are dropped."""
    vol = _random_vol_series(n=500)
    labels = label_regime_vol(vol, min_window=252, smoothing_window=5)
    assert labels.index.min() >= vol.index[252 - 1]


def test_label_regime_vol_varied_input_exercises_all_three_classes():
    vol = _random_vol_series(n=2000, seed=5)
    labels = label_regime_vol(vol, min_window=252, smoothing_window=5)
    assert set(labels.unique()) == {0, 1, 2}


def test_label_regime_vol_monotone_input_lands_current_value_above_terciles():
    """Monotone-increasing vol makes every current value the max of its
    expanding window, so every label must be 2."""
    vol = _monotone_vol_series(n=1000)
    labels = label_regime_vol(vol, min_window=252, smoothing_window=1)
    assert (labels == 2).all()


def test_label_regime_vol_smoothing_window_removes_single_day_flip():
    """A single spiked day in an otherwise-low-vol series must not retain
    its isolated label 2 after the trailing median filter."""
    dates = pd.DatetimeIndex(pd.bdate_range("2015-01-01", periods=500).values)
    vol = pd.Series(np.full(500, 0.001), index=dates)
    vol.iloc[300] = 1.0  # huge one-day spike

    labels = label_regime_vol(vol, min_window=252, smoothing_window=5)

    assert labels.loc[vol.index[300]] != 2


def test_label_regime_vol_at_t_is_invariant_to_future_mutation():
    """R0 §6.1 causality guard: the label is a decision-time feature, so the
    value at t must not change when any data after t is mutated."""
    vol = _random_vol_series(n=600, seed=3)
    labels_full = label_regime_vol(vol, min_window=252, smoothing_window=5)

    mutated = vol.copy()
    mutated.iloc[500:] = 10.0  # absurd future regime shift
    labels_mutated = label_regime_vol(mutated, min_window=252, smoothing_window=5)

    cutoff = vol.index[500]
    past = labels_full.index[labels_full.index < cutoff]
    pd.testing.assert_series_equal(labels_full.loc[past], labels_mutated.loc[past])


def test_label_regime_vol_is_free_of_look_ahead():
    """Labels on a prefix of the series must match labels from the full
    series on the entire shared index — no edge-effect exemption now that
    the smoother is trailing."""
    vol = _random_vol_series(n=1000, seed=7)

    labels_full = label_regime_vol(vol, min_window=252, smoothing_window=5)
    labels_prefix = label_regime_vol(vol.iloc[:800], min_window=252, smoothing_window=5)

    common = labels_full.index.intersection(labels_prefix.index)
    assert len(common) > 0
    pd.testing.assert_series_equal(labels_full.loc[common], labels_prefix.loc[common])


def test_label_regime_vol_empty_input_raises():
    with pytest.raises(ValueError):
        label_regime_vol(pd.Series([], dtype=float), min_window=252, smoothing_window=5)


# ---------------------------------------------------------- direction label


def test_label_direction_values_in_zero_one_two():
    close = _random_walk_close(n=800, seed=1)
    labels = label_direction(close, k=5, dead_zone_mult=0.25, vol_window=20)
    assert set(labels.unique()).issubset({0, 1, 2})
    assert labels.dtype.kind == "i"


def test_label_direction_drops_warmup_and_final_k_rows():
    """First vol_window rows have no trailing vol; last k rows have no
    resolved forward return. Both must be absent, never NaN-filled."""
    close = _random_walk_close(n=300, seed=2)
    labels = label_direction(close, k=5, dead_zone_mult=0.25, vol_window=20)

    assert labels.index.max() == close.index[-(5 + 1)]
    assert labels.index.min() >= close.index[20]
    assert not labels.isna().any()


def test_label_direction_steady_trend_is_up():
    """A steady exponential uptrend with zero noise has zero trailing vol,
    so any positive 5-day move clears the dead zone: every label is UP."""
    dates = pd.DatetimeIndex(pd.bdate_range("2015-01-01", periods=200).values)
    close = pd.Series(np.exp(np.arange(200) * 0.01), index=dates)
    labels = label_direction(close, k=5, dead_zone_mult=0.25, vol_window=20)
    assert (labels == 2).all()


def test_label_direction_constant_price_is_flat():
    dates = pd.DatetimeIndex(pd.bdate_range("2015-01-01", periods=200).values)
    close = pd.Series(np.full(200, 1.1), index=dates)
    labels = label_direction(close, k=5, dead_zone_mult=0.25, vol_window=20)
    assert (labels == 1).all()


def test_label_direction_signs_match_forward_returns():
    """UP days must have positive 5d-ahead log returns, DOWN days negative."""
    close = _random_walk_close(n=1000, seed=4)
    labels = label_direction(close, k=5, dead_zone_mult=0.25, vol_window=20)

    fwd = np.log(close.shift(-5) / close).loc[labels.index]
    assert (fwd[labels == 2] > 0).all()
    assert (fwd[labels == 0] < 0).all()


def test_label_direction_wider_dead_zone_never_shrinks_flat_share():
    close = _random_walk_close(n=1000, seed=6)
    narrow = label_direction(close, k=5, dead_zone_mult=0.25, vol_window=20)
    wide = label_direction(close, k=5, dead_zone_mult=1.0, vol_window=20)
    assert (wide == 1).sum() >= (narrow == 1).sum()


def test_label_direction_threshold_is_causal():
    """The dead-zone threshold at t must use only trailing vol. Labels on a
    prefix ending at t+k must equal labels from the full series."""
    close = _random_walk_close(n=600, seed=8)
    labels_full = label_direction(close, k=5, dead_zone_mult=0.25, vol_window=20)
    labels_prefix = label_direction(
        close.iloc[:400], k=5, dead_zone_mult=0.25, vol_window=20
    )

    common = labels_full.index.intersection(labels_prefix.index)
    assert len(common) > 0
    pd.testing.assert_series_equal(labels_full.loc[common], labels_prefix.loc[common])


def test_label_direction_empty_input_raises():
    with pytest.raises(ValueError):
        label_direction(pd.Series([], dtype=float), k=5, dead_zone_mult=0.25, vol_window=20)
