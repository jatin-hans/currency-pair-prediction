"""09: fetch CFTC Traders-in-Financial-Futures (COT) positioning → parquet.

Weekly net futures positions of "Leveraged Funds" (hedge funds / CTAs)
and "Asset Managers" in EURO FX and JAPANESE YEN contracts on the CME.
Positions are as of Tuesday and published Friday afternoon — each row is
stamped with its PUBLICATION date (+3 days) so downstream joins are
causal. No rupee contract exists in the TFF report, so USDINR has no
COT series.

Source files: https://www.cftc.gov/files/dea/history/fut_fin_txt_YYYY.zip
Output: data/raw/macro/cot_EUR.parquet, cot_JPY.parquet
"""

from __future__ import annotations

import io
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

import certifi
import pandas as pd

SSL_CTX = ssl.create_default_context(cafile=certifi.where())

ROOT = Path(__file__).resolve().parents[1]

CONTRACTS = {
    "EUR": "EURO FX - CHICAGO MERCANTILE EXCHANGE",
    "JPY": "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE",
}
YEARS = range(2015, 2027)
PUBLICATION_LAG_DAYS = 3  # positions as of Tuesday, published Friday

COLS = {
    "Market_and_Exchange_Names": "market",
    "Report_Date_as_YYYY-MM-DD": "date",
    "Lev_Money_Positions_Long_All": "lev_long",
    "Lev_Money_Positions_Short_All": "lev_short",
    "Asset_Mgr_Positions_Long_All": "am_long",
    "Asset_Mgr_Positions_Short_All": "am_short",
}


def fetch_year(year: int) -> pd.DataFrame:
    url = f"https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.4"})
    with urllib.request.urlopen(req, timeout=60, context=SSL_CTX) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".txt"))
        df = pd.read_csv(io.BytesIO(zf.read(name)), low_memory=False)
    df = df[list(COLS)].rename(columns=COLS)
    return df[df["market"].isin(CONTRACTS.values())]


def main() -> int:
    out_dir = ROOT / "data" / "raw" / "macro"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for year in YEARS:
        try:
            frames.append(fetch_year(year))
            print(f"  {year}: ok")
        except Exception as exc:  # noqa: BLE001 - a missing future year is fine
            print(f"  {year}: skipped ({exc})")
    allrows = pd.concat(frames, ignore_index=True)

    for code, market in CONTRACTS.items():
        g = allrows[allrows["market"] == market].copy()
        g["date"] = pd.to_datetime(g["date"]) + pd.Timedelta(days=PUBLICATION_LAG_DAYS)
        # Holiday-shifted reports (Monday report dates) publish LATER, never
        # earlier: roll any non-Friday stamp forward to the next Friday.
        # Caveat: the 2018-19 government shutdown suspended COT publication
        # for weeks; those reports came out months late, so these stamps
        # understate that delay.
        g["date"] += pd.to_timedelta((4 - g["date"].dt.weekday) % 7, unit="D")
        g = g.set_index("date").sort_index()
        out = pd.DataFrame(
            {
                "lev_net": g["lev_long"].astype(float) - g["lev_short"].astype(float),
                "am_net": g["am_long"].astype(float) - g["am_short"].astype(float),
            }
        )
        out.to_parquet(out_dir / f"cot_{code}.parquet")
        print(f"cot_{code}: {len(out)} weeks, {out.index[0].date()} → "
              f"{out.index[-1].date()}, latest lev_net={out['lev_net'].iloc[-1]:+.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
