"""08: fetch macro series (rates, CPI) from FRED → data/raw/macro/.

Several OECD families on FRED are discontinued, so coverage is uneven:

  rate_US  DGS3MO             3-month Treasury, daily, live
  rate_EZ  ECBDFR             ECB deposit facility rate, daily, live
  rate_JP  IR3TIB01JPM156N    3-month interbank, monthly, ~2-month lag
  rate_IN  INDIRLTLT01STM     10-YEAR gov yield (no live short rate for
                              India on FRED — long-rate proxy)
  cpi_US   CPIAUCSL           CPI index, monthly, live
  cpi_EZ   CP0000EZ19M086NEST HICP index, monthly, live
  (Japan/India CPI: no live FRED series — the value feature is EURUSD-only)

No API key needed: fredgraph.csv endpoint.
"""

from __future__ import annotations

import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SERIES = {
    "rate_US": "DGS3MO",
    "rate_EZ": "ECBDFR",
    "rate_JP": "IR3TIB01JPM156N",
    "rate_IN": "INDIRLTLT01STM",
    "cpi_US": "CPIAUCSL",
    "cpi_EZ": "CP0000EZ19M086NEST",
}


def fetch(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        raw = resp.read().decode()
    df = pd.read_csv(io.StringIO(raw))
    date_col, val_col = df.columns[0], df.columns[1]
    s = pd.Series(
        pd.to_numeric(df[val_col], errors="coerce").values,
        index=pd.to_datetime(df[date_col]),
        name=series_id,
    ).dropna()
    return s


def main() -> int:
    out_dir = ROOT / "data" / "raw" / "macro"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'name':<9}{'series':<22}{'n':>6}  {'first':<12}{'last':<12}")
    for name, sid in SERIES.items():
        s = fetch(sid)
        s.to_frame("value").to_parquet(out_dir / f"{name}.parquet")
        print(f"{name:<9}{sid:<22}{len(s):>6}  {s.index[0].date()!s:<12}"
              f"{s.index[-1].date()!s:<12}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
