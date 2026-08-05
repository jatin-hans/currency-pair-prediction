"""Walk-forward splitters with leakage assertions (spec §4.4).

Every split goes through ``assert_no_leakage``. Random splits are never
acceptable for regime data — they leak future information into the
training set.
"""

from __future__ import annotations

import pandas as pd

from regime_lab.config import WalkForwardBlock


def assert_no_leakage(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Raise if the split would leak future information into training.

    Enforces ``train.index.max() < test.index.min()`` and rejects empty
    frames — an empty frame trivially satisfies an ordering predicate but
    signals a data bug we should fail loudly on.
    """
    if len(train) == 0:
        raise AssertionError("leakage check failed: train frame is empty")
    if len(test) == 0:
        raise AssertionError("leakage check failed: test frame is empty")
    if train.index.max() >= test.index.min():
        raise AssertionError(
            "leakage check failed: "
            f"train.max={train.index.max()!r} >= test.min={test.index.min()!r}"
        )


def split_by_block(
    frame: pd.DataFrame, block: WalkForwardBlock, *, purge: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice ``frame`` into a (train, test) pair bounded by ``block``.

    ``frame`` must have a ``DatetimeIndex``. The returned frames include
    only rows whose index falls inside the block's train or test window.
    ``purge`` drops the last N train rows: with a k-day-ahead
    target, the final k train rows would otherwise see test-window prices
    through their labels — pass ``purge=k``. ``assert_no_leakage`` runs
    before the result is returned.
    """
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("split_by_block requires a DatetimeIndex")
    if purge < 0:
        raise ValueError("purge must be ≥ 0")

    train_start = pd.Timestamp(block.train_start)
    train_end = pd.Timestamp(block.train_end)
    test_start = pd.Timestamp(block.test_start)
    test_end = pd.Timestamp(block.test_end)

    train = frame.loc[(frame.index >= train_start) & (frame.index <= train_end)]
    test = frame.loc[(frame.index >= test_start) & (frame.index <= test_end)]

    if purge > 0:
        train = train.iloc[:-purge]

    assert_no_leakage(train, test)
    return train, test
