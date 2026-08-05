"""Tests for regime_lab.data.features.

Schema is spec §4.3. Invariants: no forward-fill (cross-phase invariant #2),
no look-ahead on any rolling feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lab.data.features import FEATURE_COLUMNS, build_features


def _ohlc(values: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.bdate_range(start, periods=len(values)).values)
    close = pd.Series(values, index=dates, dtype=float)
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 0},
        index=dates,
    )


def _vix_frame(values: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.bdate_range(start, periods=len(values)).values)
    return pd.DataFrame({"close": values}, index=dates)


def _long_ohlc(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.01, size=n)
    close = 1.1 * np.exp(np.cumsum(returns))
    dates = pd.DatetimeIndex(pd.bdate_range("2018-01-01", periods=n).values)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 0,
        },
        index=dates,
    )


def test_feature_columns_list_matches_spec_section_4_3():
    assert FEATURE_COLUMNS == [
        "close",
        "log_return",
        "realized_vol_5d",
        "realized_vol_20d",
        "realized_vol_60d",
        "return_z_252d",
        "dxy_return",
        "vix_level",
        "corr_dxy_60d",
        "day_of_week",
        "month",
    ]


def test_build_features_output_has_spec_schema():
    pair = _long_ohlc()
    dxy = _long_ohlc(seed=1)
    vix = _vix_frame([20.0] * len(pair), start="2018-01-01")

    features = build_features(pair, dxy, vix)

    assert list(features.columns) == FEATURE_COLUMNS
    assert features.index.name == "date"


def test_build_features_has_no_nan_anywhere():
    """Spec §4.3: drop all rows with any NaN. No forward-fill."""
    pair = _long_ohlc()
    dxy = _long_ohlc(seed=1)
    vix = _vix_frame([20.0] * len(pair), start="2018-01-01")

    features = build_features(pair, dxy, vix)

    assert not features.isna().any().any()


def test_build_features_log_return_matches_manual_calc():
    pair = _ohlc([1.0, 1.01, 1.02, 1.03, 1.04] * 80)
    dxy = _ohlc([100.0, 100.5, 101.0, 101.5, 102.0] * 80)
    vix = _vix_frame([20.0] * len(pair))

    features = build_features(pair, dxy, vix)

    expected = np.log(pair["close"] / pair["close"].shift(1)).loc[features.index]
    np.testing.assert_allclose(features["log_return"].to_numpy(), expected.to_numpy())


def test_build_features_day_of_week_and_month_have_valid_ranges():
    pair = _long_ohlc()
    dxy = _long_ohlc(seed=1)
    vix = _vix_frame([20.0] * len(pair), start="2018-01-01")

    features = build_features(pair, dxy, vix)

    assert set(features["day_of_week"].unique()).issubset({0, 1, 2, 3, 4})
    assert features["month"].between(1, 12).all()


def test_build_features_realized_vol_is_rolling_std_of_log_return():
    pair = _long_ohlc(n=300)
    dxy = _long_ohlc(n=300, seed=2)
    vix = _vix_frame([20.0] * len(pair), start="2018-01-01")

    features = build_features(pair, dxy, vix)

    full_log_ret = np.log(pair["close"] / pair["close"].shift(1))
    expected_5d = full_log_ret.rolling(5).std().loc[features.index]

    np.testing.assert_allclose(
        features["realized_vol_5d"].to_numpy(),
        expected_5d.to_numpy(),
        rtol=1e-10,
    )


def test_build_features_rejects_forward_fill_by_dropping_missing_dxy_rows():
    """Cross-phase invariant #2: no forward-fill. Missing DXY rows drop.

    Fixture uses n=700 so that after taking the intersection of pair and
    the partial DXY (600 rows common) there is still enough history for
    the 252-day z-score window.
    """
    pair = _long_ohlc(n=700)
    dxy_partial = pair.copy().iloc[100:]  # first 100 DXY rows missing

    vix = _vix_frame([20.0] * len(pair), start="2018-01-01")
    features = build_features(pair, dxy_partial, vix)

    assert len(features) > 0
    assert features.index.min() >= dxy_partial.index.min()


def test_build_features_is_free_of_look_ahead_for_rolling_features():
    """Re-computing features on a prefix must yield identical rows for that prefix."""
    pair = _long_ohlc(n=500)
    dxy = _long_ohlc(n=500, seed=3)
    vix = _vix_frame([20.0] * len(pair), start="2018-01-01")

    full = build_features(pair, dxy, vix)
    prefix = build_features(pair.iloc[:400], dxy.iloc[:400], vix.iloc[:400])

    common = full.index.intersection(prefix.index)
    assert len(common) > 0
    pd.testing.assert_frame_equal(
        full.loc[common][FEATURE_COLUMNS],
        prefix.loc[common][FEATURE_COLUMNS],
    )


def test_build_features_fails_loudly_on_empty_input():
    with pytest.raises(ValueError):
        build_features(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
