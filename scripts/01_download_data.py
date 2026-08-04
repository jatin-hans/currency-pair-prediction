"""Download raw OHLC for the three FX pairs plus DXY and VIX.

Uses ``regime_lab.data.loaders.fetch_ohlc`` (cache-first). Re-running is
idempotent: tickers whose parquet already exists in ``data/raw/`` are
served from cache. Pass ``--refresh`` to force a re-download.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make ``src/`` importable when this script is run directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime_lab.config import Settings  # noqa: E402
from regime_lab.data.loaders import fetch_ohlc  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore the local cache and re-download every ticker.",
    )
    args = parser.parse_args()

    settings = Settings()
    raw_dir = ROOT / settings.data_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    tickers = {**settings.pairs, **settings.auxiliary_tickers}

    print(f"Fetching {len(tickers)} tickers into {raw_dir}")
    print(
        f"  range: {settings.date_start.isoformat()} "
        f"→ {settings.date_end.isoformat()}"
    )
    print()
    print(f"{'name':<10} {'ticker':<12} {'rows':>6} {'first_date':<12} "
          f"{'last_date':<12} {'elapsed_s':>9}")
    print("-" * 65)

    for name, ticker in tickers.items():
        t0 = time.perf_counter()
        frame = fetch_ohlc(
            ticker,
            start=settings.date_start,
            end=settings.date_end,
            cache_dir=raw_dir,
            force_refresh=args.refresh,
        )
        dt = time.perf_counter() - t0
        print(
            f"{name:<10} {ticker:<12} {len(frame):>6} "
            f"{frame.index.min().date().isoformat():<12} "
            f"{frame.index.max().date().isoformat():<12} "
            f"{dt:>9.2f}"
        )

    print()
    print(f"Done. Cache files in {raw_dir}:")
    for path in sorted(raw_dir.glob("*.parquet")):
        print(f"  {path.name:<20} {path.stat().st_size/1024:>8.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
