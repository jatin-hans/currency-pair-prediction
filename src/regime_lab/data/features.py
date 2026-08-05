"""Feature engineering for the FX regime dataset (spec §4.3).

``build_features`` takes raw OHLC frames for a pair plus the auxiliary DXY
and VIX series, and returns the full feature table with NaN rows dropped.

Invariant: no forward-fill. Rows where any feature is NaN after
construction are dropped; nothing is interpolated across gaps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS: list[str] = [
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


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def build_features(
    pair_ohlc: pd.DataFrame,
    dxy_ohlc: pd.DataFrame,
    vix_ohlc: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the feature table for one currency pair (spec §4.3)."""
    if pair_ohlc.empty or dxy_ohlc.empty or vix_ohlc.empty:
        raise ValueError("build_features requires non-empty OHLC frames")

    # Cross-calendar alignment: FX, DXY, and VIX trade on overlapping but
    # non-identical calendars (US holidays, FX-only days, VIX half-days).
    # Spec §4.3 forbids forward-fill, so we take the intersection of the
    # three calendars and compute every feature on that common index.
    # Rows that exist for only a subset of the three feeds are dropped,
    # not interpolated.
    pair = pair_ohlc.sort_index()
    dxy = dxy_ohlc.sort_index()
    vix = vix_ohlc.sort_index()
    common_index = pair.index.intersection(dxy.index).intersection(vix.index)
    pair = pair.loc[common_index]
    dxy = dxy.loc[common_index]
    vix = vix.loc[common_index]

    # Yahoo stamps FX daily closes one bar EARLIER than the DXY/VIX index
    # closes — for EURUSD, corr(dxy_return_t, fx_return_{t+1}) ≈ −0.87
    # versus −0.10 contemporaneous, i.e. same-day index values would sit
    # inside the return being predicted. Lag both one trading day so every
    # DXY/VIX value is known strictly before the FX close it joins to.
    dxy = dxy.shift(1)
    vix = vix.shift(1)

    features = pd.DataFrame(index=common_index)
    features.index.name = "date"

    features["close"] = pair["close"].astype(float)
    features["log_return"] = np.log(pair["close"] / pair["close"].shift(1))

    features["realized_vol_5d"] = features["log_return"].rolling(5).std()
    features["realized_vol_20d"] = features["log_return"].rolling(20).std()
    features["realized_vol_60d"] = features["log_return"].rolling(60).std()
    features["return_z_252d"] = _rolling_zscore(features["log_return"], 252)

    features["dxy_return"] = np.log(dxy["close"] / dxy["close"].shift(1))
    features["vix_level"] = vix["close"].astype(float)

    features["corr_dxy_60d"] = (
        features["log_return"].rolling(60).corr(features["dxy_return"])
    )

    features["day_of_week"] = features.index.dayofweek.astype("int64")
    features["month"] = features.index.month.astype("int64")

    # Spec §4.3: drop rows with any NaN left over (rolling-window warmup).
    features = features.dropna(how="any")

    return features[FEATURE_COLUMNS]
