"""yfinance loaders with a local parquet cache.

Tests mock ``_download_from_yfinance``; production code hits the network.
Cache layout: ``{cache_dir}/{sanitized_ticker}.parquet``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

_NON_FILENAME_CHARS = str.maketrans({"=": "_", "^": "_", ".": "_", "/": "_", ":": "_"})


def _sanitize_ticker(ticker: str) -> str:
    """Map a yfinance ticker to a filesystem-safe filename stem."""
    return ticker.translate(_NON_FILENAME_CHARS)


def _download_from_yfinance(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Fetch daily OHLC from yfinance. Seam for tests to monkeypatch."""
    import yfinance as yf

    raw = yf.download(
        ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()

    # yfinance occasionally returns a MultiIndex on columns when one ticker is
    # passed; flatten so downstream code can rely on simple column names.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    rename = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    raw = raw.rename(columns=rename)
    raw.index.name = "date"
    return raw


def fetch_ohlc(
    ticker: str,
    *,
    start: date,
    end: date,
    cache_dir: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return daily OHLC for ``ticker`` between ``start`` and ``end``.

    Cache-first: if ``{cache_dir}/{sanitize(ticker)}.parquet`` exists and
    ``force_refresh`` is False, the cache is returned as-is. Otherwise the
    frame is downloaded, written to the cache path, and returned.

    Raises ``ValueError`` if the download is empty so the caller notices
    early rather than polluting the cache with an empty parquet.
    """
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"{_sanitize_ticker(ticker)}.parquet"

    if cache_path.exists() and not force_refresh:
        return pd.read_parquet(cache_path)

    frame = _download_from_yfinance(ticker, start, end)
    if frame is None or frame.empty:
        raise ValueError(f"download returned empty frame for ticker {ticker!r}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path)
    return frame
