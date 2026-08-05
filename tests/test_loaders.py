"""Tests for regime_lab.data.loaders.

The public API is cache-first: if a parquet exists in ``cache_dir``,
``fetch_ohlc`` returns it without touching the network. This test suite
monkeypatches the yfinance seam so no real network I/O happens.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from regime_lab.data import loaders


def _synthetic_ohlc(start: str = "2020-01-01", end: str = "2020-01-31") -> pd.DataFrame:
    # freq=None so the frame survives a parquet round-trip byte-for-byte.
    dates = pd.DatetimeIndex(pd.bdate_range(start, end).values)
    return pd.DataFrame(
        {
            "open": 1.10,
            "high": 1.11,
            "low": 1.09,
            "close": 1.10 + 0.001 * pd.Series(range(len(dates)), index=dates),
            "volume": 0,
        },
        index=dates,
    )


def test_fetch_ohlc_reads_cache_when_present(tmp_path: Path, monkeypatch):
    cache_dir = tmp_path / "raw"
    cache_dir.mkdir()
    frame = _synthetic_ohlc()
    cache_path = cache_dir / "EURUSD_X.parquet"
    frame.to_parquet(cache_path)

    def _should_not_be_called(*args, **kwargs):
        raise RuntimeError("network path should not be taken when cache exists")

    monkeypatch.setattr(loaders, "_download_from_yfinance", _should_not_be_called)

    result = loaders.fetch_ohlc(
        "EURUSD=X", start=date(2020, 1, 1), end=date(2020, 1, 31), cache_dir=cache_dir
    )

    pd.testing.assert_frame_equal(result, frame)


def test_fetch_ohlc_downloads_and_caches_when_missing(tmp_path: Path, monkeypatch):
    cache_dir = tmp_path / "raw"
    cache_dir.mkdir()
    synthetic = _synthetic_ohlc()
    call_log: list[tuple[str, date, date]] = []

    def fake_download(ticker, start, end):
        call_log.append((ticker, start, end))
        return synthetic

    monkeypatch.setattr(loaders, "_download_from_yfinance", fake_download)

    result = loaders.fetch_ohlc(
        "EURUSD=X", start=date(2020, 1, 1), end=date(2020, 1, 31), cache_dir=cache_dir
    )

    assert call_log == [("EURUSD=X", date(2020, 1, 1), date(2020, 1, 31))]
    pd.testing.assert_frame_equal(result, synthetic)
    cache_path = cache_dir / "EURUSD_X.parquet"
    assert cache_path.exists()


def test_fetch_ohlc_force_refresh_ignores_cache(tmp_path: Path, monkeypatch):
    cache_dir = tmp_path / "raw"
    cache_dir.mkdir()
    stale = _synthetic_ohlc()
    cache_path = cache_dir / "EURUSD_X.parquet"
    stale.to_parquet(cache_path)

    fresh = _synthetic_ohlc(start="2020-02-01", end="2020-02-29")
    monkeypatch.setattr(loaders, "_download_from_yfinance", lambda *a, **k: fresh)

    result = loaders.fetch_ohlc(
        "EURUSD=X",
        start=date(2020, 1, 1),
        end=date(2020, 2, 29),
        cache_dir=cache_dir,
        force_refresh=True,
    )

    pd.testing.assert_frame_equal(result, fresh)
    refreshed = pd.read_parquet(cache_path)
    pd.testing.assert_frame_equal(refreshed, fresh)


def test_fetch_ohlc_rejects_empty_download(tmp_path: Path, monkeypatch):
    cache_dir = tmp_path / "raw"
    cache_dir.mkdir()
    monkeypatch.setattr(
        loaders,
        "_download_from_yfinance",
        lambda *a, **k: pd.DataFrame(),
    )

    with pytest.raises(ValueError, match="empty"):
        loaders.fetch_ohlc(
            "BADTICKER",
            start=date(2020, 1, 1),
            end=date(2020, 1, 31),
            cache_dir=cache_dir,
        )


def test_fetch_ohlc_sanitizes_ticker_for_filename(tmp_path: Path, monkeypatch):
    """Tickers like '^VIX' or 'DX-Y.NYB' must not produce invalid filenames."""
    cache_dir = tmp_path / "raw"
    cache_dir.mkdir()
    monkeypatch.setattr(loaders, "_download_from_yfinance", lambda *a, **k: _synthetic_ohlc())

    loaders.fetch_ohlc(
        "^VIX", start=date(2020, 1, 1), end=date(2020, 1, 31), cache_dir=cache_dir
    )
    loaders.fetch_ohlc(
        "DX-Y.NYB", start=date(2020, 1, 1), end=date(2020, 1, 31), cache_dir=cache_dir
    )

    files = sorted(p.name for p in cache_dir.iterdir())
    assert files == ["DX-Y_NYB.parquet", "_VIX.parquet"]
