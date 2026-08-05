"""Tests for regime_lab.data.splits.

Walk-forward splits are spec §4.4. The leakage assertion is non-negotiable
per the phased plan's cross-phase invariant #1 — it gets its own test.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from regime_lab.config import Settings, WalkForwardBlock
from regime_lab.data.splits import assert_no_leakage, split_by_block


def _sample_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2015-01-01", "2026-04-18")
    return pd.DataFrame({"log_return": [0.0] * len(dates)}, index=dates)


def test_split_by_block_returns_train_and_test_dataframes():
    frame = _sample_frame()
    block = Settings().walk_forward_blocks[0]

    train, test = split_by_block(frame, block)

    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)


def test_split_by_block_train_bounded_by_block_train_window():
    frame = _sample_frame()
    block = Settings().walk_forward_blocks[0]

    train, _ = split_by_block(frame, block)

    assert train.index.min().date() >= block.train_start
    assert train.index.max().date() <= block.train_end


def test_split_by_block_test_bounded_by_block_test_window():
    frame = _sample_frame()
    block = Settings().walk_forward_blocks[0]

    _, test = split_by_block(frame, block)

    assert test.index.min().date() >= block.test_start
    assert test.index.max().date() <= block.test_end


def test_split_by_block_train_and_test_are_non_empty_for_every_block():
    frame = _sample_frame()

    for block in Settings().walk_forward_blocks:
        train, test = split_by_block(frame, block)
        assert len(train) > 0, f"block {block.block_id} has empty train"
        assert len(test) > 0, f"block {block.block_id} has empty test"


def test_split_by_block_guarantees_train_before_test():
    """Spec §4.4: every split must satisfy train.max() < test.min()."""
    frame = _sample_frame()

    for block in Settings().walk_forward_blocks:
        train, test = split_by_block(frame, block)
        assert train.index.max() < test.index.min()


def test_split_by_block_purge_drops_last_k_train_rows():
    """R0 §6.2: purge removes exactly the last k train rows."""
    frame = _sample_frame()
    block = Settings().walk_forward_blocks[0]

    train_full, _ = split_by_block(frame, block)
    train_purged, _ = split_by_block(frame, block, purge=5)

    assert len(train_purged) == len(train_full) - 5
    pd.testing.assert_index_equal(train_purged.index, train_full.index[:-5])


def test_split_by_block_purge_keeps_forward_target_out_of_test_window():
    """With a k-day-ahead target and purge=k, the target of the last train
    row must resolve strictly before the test window begins."""
    frame = _sample_frame()
    k = 5

    for block in Settings().walk_forward_blocks:
        train, test = split_by_block(frame, block, purge=k)
        last_train_pos = frame.index.get_indexer([train.index.max()])[0]
        target_date = frame.index[last_train_pos + k]
        assert target_date < test.index.min(), f"block {block.block_id} leaks"


def test_split_by_block_purge_zero_is_default_and_unchanged():
    frame = _sample_frame()
    block = Settings().walk_forward_blocks[0]

    train_default, _ = split_by_block(frame, block)
    train_zero, _ = split_by_block(frame, block, purge=0)

    pd.testing.assert_frame_equal(train_default, train_zero)


def test_assert_no_leakage_passes_on_ordered_frames():
    dates_train = pd.bdate_range("2015-01-01", "2017-06-30")
    dates_test = pd.bdate_range("2017-07-01", "2018-06-30")
    train = pd.DataFrame(index=dates_train)
    test = pd.DataFrame(index=dates_test)

    assert_no_leakage(train, test)  # must not raise


def test_assert_no_leakage_fires_on_overlapping_frames():
    """The cross-phase invariant #1: leakage assertion is non-negotiable."""
    dates_train = pd.bdate_range("2015-01-01", "2017-12-31")
    dates_test = pd.bdate_range("2017-06-01", "2018-06-30")  # overlap
    train = pd.DataFrame(index=dates_train)
    test = pd.DataFrame(index=dates_test)

    with pytest.raises(AssertionError, match="leakage"):
        assert_no_leakage(train, test)


def test_assert_no_leakage_fires_on_reversed_frames():
    dates_train = pd.bdate_range("2020-01-01", "2020-12-31")
    dates_test = pd.bdate_range("2019-01-01", "2019-12-31")  # test before train
    train = pd.DataFrame(index=dates_train)
    test = pd.DataFrame(index=dates_test)

    with pytest.raises(AssertionError, match="leakage"):
        assert_no_leakage(train, test)


def test_assert_no_leakage_fires_when_either_frame_is_empty():
    empty = pd.DataFrame(index=pd.DatetimeIndex([]))
    non_empty = pd.DataFrame(index=pd.bdate_range("2020-01-01", "2020-12-31"))

    with pytest.raises(AssertionError):
        assert_no_leakage(empty, non_empty)
    with pytest.raises(AssertionError):
        assert_no_leakage(non_empty, empty)


def test_custom_block_train_after_test_fails_leakage_check():
    """Sanity: a deliberately-reversed custom block fails leakage."""
    frame = _sample_frame()
    bad = WalkForwardBlock(
        block_id=99,
        train_start=date(2021, 1, 1),
        train_end=date(2022, 12, 31),
        test_start=date(2020, 1, 1),
        test_end=date(2020, 12, 31),
    )

    with pytest.raises(AssertionError, match="leakage"):
        split_by_block(frame, bad)
