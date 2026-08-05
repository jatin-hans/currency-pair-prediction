"""Labels for the direction task.

``label_regime_vol`` — volatility-state feature: rolling tercile of
``realized_vol_20d`` on an expanding window ≥252 observations, smoothed
with a trailing median filter. Trailing (not centered) keeps it causal:
the state is a decision-time feature, so the value at t may only use
data up to t.

``label_direction`` — the prediction target: direction of the k-day-ahead
log return, {0: DOWN, 1: FLAT, 2: UP}, where FLAT means the move is inside
a dead zone of ``dead_zone_mult × trailing σ(vol_window) × √k``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DIRECTION_NAMES = {0: "DOWN", 1: "FLAT", 2: "UP"}
STATE_NAMES = {0: "calm", 1: "normal", 2: "turbulent"}


def label_regime_vol(
    vol: pd.Series,
    *,
    min_window: int = 252,
    smoothing_window: int = 5,
) -> pd.Series:
    """Assign volatility state {0, 1, 2} to each row of a vol series.

    Step 1: for each row t with ≥ ``min_window`` observations in [0:t],
    compute the 1/3 and 2/3 quantiles of the expanding window and bucket
    the current value.

    Step 2: apply a *trailing* ``smoothing_window`` median filter so that
    single-day flips do not create spurious state changes. The filter at t
    sees rows (t-w+1 … t) only — never the future.

    Returns an integer series aligned to the input index, covering only
    rows that had enough history to label. Earlier rows are dropped.
    """
    if len(vol) == 0:
        raise ValueError("label_regime_vol requires a non-empty series")
    if smoothing_window < 1:
        raise ValueError("smoothing_window must be ≥ 1")

    vol = vol.astype(float)

    lower = vol.expanding(min_periods=min_window).quantile(1.0 / 3.0)
    upper = vol.expanding(min_periods=min_window).quantile(2.0 / 3.0)

    raw = pd.Series(np.nan, index=vol.index, dtype=float)
    raw[vol < lower] = 0.0
    raw[(vol >= lower) & (vol <= upper)] = 1.0
    raw[vol > upper] = 2.0

    # Drop the insufficient-history prefix BEFORE smoothing, so the median
    # filter cannot drag valid labels backward into the unlabelled region.
    raw = raw.dropna()

    if smoothing_window > 1:
        smoothed = raw.rolling(smoothing_window, min_periods=1).median()
    else:
        smoothed = raw

    labels = smoothed.round().astype("int64")
    labels.name = "regime_label_vol"
    return labels


def label_direction(
    close: pd.Series,
    *,
    k: int = 5,
    dead_zone_mult: float = 0.25,
    vol_window: int = 20,
) -> pd.Series:
    """Direction of the k-day-ahead log return: {0: DOWN, 1: FLAT, 2: UP}.

    The dead-zone threshold at t is ``dead_zone_mult × σ × √k`` where σ is
    the trailing ``vol_window``-day stdev of daily log returns known at t
    (the threshold never uses future vol). FLAT means
    ``|k-day log return| ≤ threshold``.

    Rows without a trailing vol (warmup) or without a resolved forward
    return (final k rows) are dropped, never NaN-filled.
    """
    if len(close) == 0:
        raise ValueError("label_direction requires a non-empty series")
    if k < 1:
        raise ValueError("k must be ≥ 1")

    log_close = np.log(close.astype(float))
    fwd_return = log_close.shift(-k) - log_close
    trailing_vol = log_close.diff().rolling(vol_window).std()
    threshold = dead_zone_mult * trailing_vol * np.sqrt(k)

    labels = pd.Series(np.nan, index=close.index, dtype=float)
    valid = fwd_return.notna() & threshold.notna()
    labels[valid & (fwd_return.abs() <= threshold)] = 1.0
    labels[valid & (fwd_return > threshold)] = 2.0
    labels[valid & (fwd_return < -threshold)] = 0.0

    labels = labels.dropna().astype("int64")
    labels.name = "direction_label"
    return labels
