"""14: period-average exchange-rate forecasting.

At the last business day of a month, estimate the AVERAGE exchange rate
over the coming month. Baselines: prev_avg (last month's average carried
forward), spot (latest close carried forward — the correct no-change
benchmark for a period-average target, per RBA RDP 2025-09), ar (recursive
AR(1) on daily log returns, expected path averaged over 21 days).
Output: tables/period_average.csv + outputs/period_average.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime_lab.config import Settings  # noqa: E402
from regime_lab.data.loaders import _sanitize_ticker  # noqa: E402

AR_WIN, FWD_DAYS = 756, 21


def forecasts_for_pair(close: pd.Series) -> pd.DataFrame:
    monthly_avg = close.resample("ME").mean()
    month_end_close = close.resample("ME").last()
    rows = []
    r = np.log(close).diff().dropna()
    for i in range(1, len(monthly_avg) - 1):
        origin = monthly_avg.index[i]
        if close.index[-1] < monthly_avg.index[i + 1]:
            break  # target month not fully observed
        hist = r[r.index <= origin]
        if len(hist) < AR_WIN:
            continue
        actual = float(monthly_avg.iloc[i + 1])
        spot = float(month_end_close.iloc[i])
        prev_avg = float(monthly_avg.iloc[i])
        h = hist.iloc[-AR_WIN:]
        mu = float(h.mean())
        phi = float(h.autocorr(lag=1))
        r_last = float(h.iloc[-1])
        path = np.cumsum([mu + (phi ** k) * (r_last - mu) for k in range(1, FWD_DAYS + 1)])
        ar = float(np.mean(spot * np.exp(path)))
        rows.append({"origin": origin.date().isoformat(),
                     "actual": actual, "prev_avg": prev_avg,
                     "spot": spot, "ar": ar})
    return pd.DataFrame(rows)


def main() -> int:
    settings = Settings()
    payload, table = {}, []
    for pair, ticker in settings.pairs.items():
        df = pd.read_parquet(ROOT / settings.data_root / "raw"
                             / f"{_sanitize_ticker(ticker)}.parquet")
        close = df["close"].astype(float).sort_index().dropna()
        fc = forecasts_for_pair(close)
        n = len(fc)
        half = n // 2
        for model in ("prev_avg", "spot", "ar"):
            err = (fc[model] - fc["actual"]) / fc["actual"] * 100
            table.append({
                "pair": pair, "model": model, "n": n,
                "mae_pct": round(float(err.abs().mean()), 3),
                "rmse_pct": round(float(np.sqrt((err ** 2).mean())), 3),
                "mae_pct_h1": round(float(err.abs().iloc[:half].mean()), 3),
                "mae_pct_h2": round(float(err.abs().iloc[half:].mean()), 3),
            })
        payload[pair] = {
            "origins": fc["origin"].tolist(),
            "actual": [round(v, 5) for v in fc["actual"]],
            "spot": [round(v, 5) for v in fc["spot"]],
            "prev_avg": [round(v, 5) for v in fc["prev_avg"]],
            "ar": [round(v, 5) for v in fc["ar"]],
        }
        print(f"  {pair}: {n} months")

    t = pd.DataFrame(table)
    t.to_csv(ROOT / "outputs" / "case_study" / "tables" / "period_average.csv",
             index=False)
    (ROOT / "outputs" / "period_average.json").write_text(
        json.dumps(payload, allow_nan=False))
    print(t.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
