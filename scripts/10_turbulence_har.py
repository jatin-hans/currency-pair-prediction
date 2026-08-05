"""10: volatility forecasting — realised vol over the next week/month.

Target: log RV over the next 5/21 trading days from daily closes (a
daily-return proxy, not intraday RV). Models: rw (current RV carried
forward), har (Corsi 2009 on daily/weekly/monthly RV components), har_ml
(HAR + lagged exogenous features in one gradient-boosting regressor).
Walk-forward, expanding window; metrics QLIKE, MAE/RMSE on log RV.
Output: outputs/turbulence_har.json, outputs/case_study/tables/turbulence_har.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime_lab.config import Settings  # noqa: E402
from regime_lab.data.loaders import _sanitize_ticker  # noqa: E402

EPS = 1e-8
HORIZONS = {"1w": dict(fwd=5, refit=13, min_train=150, med_win=252),
            "1m": dict(fwd=21, refit=3, min_train=40, med_win=252)}


def load_close(settings: Settings, ticker: str) -> pd.Series:
    df = pd.read_parquet(ROOT / settings.data_root / "raw"
                         / f"{_sanitize_ticker(ticker)}.parquet")
    return df["close"].astype(float).sort_index().dropna()


def build_frame(pair_close, dxy, vix) -> pd.DataFrame:
    idx = pair_close.index.intersection(dxy.index).intersection(vix.index)
    c = pair_close.loc[idx]
    r = np.log(c).diff()
    f = pd.DataFrame(index=idx)
    f["r2"] = r ** 2
    f["rv_d"] = r.abs()
    f["rv_w"] = np.sqrt(f["r2"].rolling(5).mean())
    f["rv_m"] = np.sqrt(f["r2"].rolling(21).mean())
    # Exogenous features, lagged one trading day (same stamp-alignment fix
    # as the direction study: Yahoo FX closes precede the index closes).
    f["vix"] = vix.loc[idx].shift(1)
    f["dxy_5d"] = np.log(dxy.loc[idx]).diff(5).shift(1)
    f["shock"] = f["rv_d"] / (f["rv_m"] + EPS)
    mom_fast, mom_slow = np.log(c).diff(21), np.log(c).diff(252)
    f["mom_agree"] = (np.sign(mom_fast) == np.sign(mom_slow)).astype(float)
    return f


def eval_horizon(f: pd.DataFrame, cfg: dict, seed: int) -> dict:
    fwd = cfg["fwd"]
    target = np.sqrt(f["r2"].shift(-1).rolling(fwd).mean().shift(-(fwd - 1)))
    # Origins: every fwd-th trading day → non-overlapping target windows.
    valid = f.dropna().index.intersection(target.dropna().index)
    origins = valid[::fwd]
    y = np.log(target.loc[origins] + EPS)
    har_cols = ["rv_d", "rv_w", "rv_m"]
    exog = ["vix", "dxy_5d", "shock", "mom_agree"]
    X_har = np.log(f.loc[origins, har_cols] + EPS)
    X_ml = pd.concat([X_har, f.loc[origins, exog]], axis=1)
    rw = np.log(f.loc[origins, "rv_w" if fwd == 5 else "rv_m"] + EPS)

    preds = {"rw": rw.to_numpy(), "har": np.full(len(y), np.nan),
             "har_ml": np.full(len(y), np.nan)}
    for start in range(cfg["min_train"], len(y), cfg["refit"]):
        end = min(start + cfg["refit"], len(y))
        tr = slice(0, start)
        ols = LinearRegression().fit(X_har.iloc[tr], y.iloc[tr])
        preds["har"][start:end] = ols.predict(X_har.iloc[start:end])
        gbr = GradientBoostingRegressor(
            n_estimators=200, max_depth=2, learning_rate=0.05,
            subsample=0.8, random_state=seed,
        ).fit(X_ml.iloc[tr], y.iloc[tr])
        preds["har_ml"][start:end] = gbr.predict(X_ml.iloc[start:end])

    test = ~np.isnan(preds["har"])
    y_t = y.to_numpy()[test]
    dates = [d.date().isoformat() for d in origins[test]]
    # Trailing-median threshold for the secondary class view (strictly
    # trailing, computed on the daily rv series then sampled at origins).
    med = f["rv_w" if fwd == 5 else "rv_m"].rolling(
        cfg["med_win"], min_periods=60).median().loc[origins].to_numpy()[test]
    actual_hi = np.exp(y_t) > med

    out = {"n": int(test.sum()), "dates": dates,
           "actual": [round(float(v), 4) for v in np.exp(y_t) * np.sqrt(252)],
           "models": {}}
    for k in ("rw", "har", "har_ml"):
        p = preds[k][test]
        h, s = np.exp(2 * p), np.exp(2 * y_t)  # variance forecast / realised
        qlike = float(np.mean(s / h - np.log(s / h) - 1))
        pred_hi = np.exp(p) > med
        tp = int((pred_hi & actual_hi).sum())
        out["models"][k] = {
            "qlike": round(qlike, 4),
            "mae_log": round(float(np.mean(np.abs(p - y_t))), 4),
            "rmse_log": round(float(np.sqrt(np.mean((p - y_t) ** 2))), 4),
            "hi_precision": round(tp / max(int(pred_hi.sum()), 1), 3),
            "hi_recall": round(tp / max(int(actual_hi.sum()), 1), 3),
            "hi_calls": int(pred_hi.sum()),
            "hi_actual": int(actual_hi.sum()),
            "series": [round(float(v), 4) for v in np.exp(p) * np.sqrt(252)],
        }
    return out


def main() -> int:
    settings = Settings()
    dxy = load_close(settings, settings.auxiliary_tickers["DXY"])
    vix = load_close(settings, settings.auxiliary_tickers["VIX"])
    payload, rows = {}, []
    for pair, ticker in settings.pairs.items():
        f = build_frame(load_close(settings, ticker), dxy, vix)
        payload[pair] = {}
        for hkey, cfg in HORIZONS.items():
            res = eval_horizon(f, cfg, settings.seed)
            payload[pair][hkey] = res
            for mk, m in res["models"].items():
                rows.append({"pair": pair, "horizon": hkey, "model": mk,
                             "n": res["n"], **{k: v for k, v in m.items()
                                               if k != "series"}})
            best = min(res["models"], key=lambda k: res["models"][k]["qlike"])
            print(f"  {pair} {hkey}: n={res['n']}, best by QLIKE: {best} "
                  f"({res['models'][best]['qlike']})")

    (ROOT / "outputs" / "turbulence_har.json").write_text(
        json.dumps(payload, allow_nan=False))
    pd.DataFrame(rows).to_csv(
        ROOT / "outputs" / "case_study" / "tables" / "turbulence_har.csv",
        index=False)
    print("Wrote outputs/turbulence_har.json and tables/turbulence_har.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
