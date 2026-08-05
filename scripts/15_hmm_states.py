"""15: export the HMM state diagnostic.

Backs the page's claim that the HMM's states are volatility states with no
directional information: per pair, fit a 3-state Gaussian HMM on daily log
returns (same seed and state count as the dashboard tier; more EM
iterations since this runs once), then report each
state's occupancy, mean, vol, and state-conditional direction frequencies
against the marginal. Output: outputs/case_study/tables/hmm_states.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hmmlearn.hmm import GaussianHMM  # noqa: E402

from regime_lab.config import Settings  # noqa: E402
from regime_lab.data.loaders import _sanitize_ticker  # noqa: E402


def main() -> int:
    settings = Settings()
    rows = []
    for pair, ticker in settings.pairs.items():
        close = pd.read_parquet(
            ROOT / settings.data_root / "raw" / f"{_sanitize_ticker(ticker)}.parquet"
        )["close"].astype(float).sort_index().dropna()
        r = np.log(close).diff().dropna()
        # Direction label with the same dead-zone recipe as the dashboard.
        thr = settings.dead_zone_mult * r.rolling(20).std()
        fwd = r.shift(-1)
        lab = pd.Series(np.where(fwd.abs() <= thr, 1,
                        np.where(fwd > thr, 2, 0)), index=r.index, dtype=float)
        lab[fwd.isna() | thr.isna()] = np.nan

        hmm = GaussianHMM(n_components=3, covariance_type="diag",
                          n_iter=200, random_state=settings.seed)
        X = r.to_numpy().reshape(-1, 1)
        hmm.fit(X)
        states = hmm.predict(X)
        marg = lab.value_counts(normalize=True).sort_index()
        for s in range(3):
            mask = (states == s) & lab.notna().to_numpy()
            freq = lab[mask].value_counts(normalize=True).sort_index()
            rows.append({
                "pair": pair, "state": s,
                "occupancy": round(float(mask.mean()), 3),
                "mean_ret_bp": round(float(r[states == s].mean() * 1e4), 2),
                "vol_bp": round(float(r[states == s].std() * 1e4), 1),
                "p_up": round(float(freq.get(2.0, 0.0)), 3),
                "p_down": round(float(freq.get(0.0, 0.0)), 3),
                "p_up_marginal": round(float(marg.get(2.0, 0.0)), 3),
                "p_down_marginal": round(float(marg.get(0.0, 0.0)), 3),
                "up_dev_pp": round(100 * float(freq.get(2.0, 0.0) - marg.get(2.0, 0.0)), 1),
            })
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "outputs" / "case_study" / "tables" / "hmm_states.csv",
              index=False)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
