"""13: FOMC-stance feature ablation on USDINR weekly direction.

Logistic and calibrated gradient boosting, each with and without the
stance features, on identical walk-forward observations. Stance features
join with a +1-day availability lag (statements land 2pm ET, after the
Yahoo FX close stamp). RBI statements are absent (portal not enumerable),
so this covers the FOMC (dollar-leg) half only.
Output: outputs/case_study/tables/stance_ablation.csv + console summary.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime_lab.config import Settings  # noqa: E402
from regime_lab.models import _expand_proba, fit_calibrated, make_b1, make_b2  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "dash", ROOT / "scripts" / "07_build_dashboard.py")
dash = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dash)

PAIR, HKEY = "USDINR", "1w"
REFIT_EVERY = 4
MIN_TRAIN = 150
BLOCK, DRAWS = 8, 2000


def stance_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    st = pd.read_parquet(ROOT / "data" / "raw" / "fomc" / "stance.parquet")
    st = st.sort_index()
    avail = st.index + pd.Timedelta(days=1)  # publication → usable next FX day
    score = pd.Series(st["score"].to_numpy(), index=avail)
    chg = score.diff()
    surprise = score - score.rolling(4).mean().shift(1)
    f = pd.DataFrame(index=index)
    f["fomc_score"] = dash._asof(score, index)
    f["fomc_chg"] = dash._asof(chg, index)
    f["fomc_surprise"] = dash._asof(surprise, index)
    last_date = pd.Series(avail, index=avail)
    f["fomc_days"] = (index - dash._asof(last_date, index)).dt.days.astype(float)
    f["fomc_decay"] = f["fomc_score"] * np.exp(-f["fomc_days"] / 45.0)
    return f


def walk_forward(frame, feats, factory, calibrated, seed):
    y_all = frame["direction_label"]
    briers, hits, acts, preds = [], [], [], []
    model = None
    eval_pos = [p for p in range(MIN_TRAIN, len(frame)) if not np.isnan(y_all.iloc[p])]
    for i, p in enumerate(eval_pos):
        if i % REFIT_EVERY == 0:
            train = frame.iloc[:p]
            train = train[train["direction_label"].notna()]
            if calibrated:
                model = fit_calibrated(factory, train[feats],
                                       train["direction_label"].astype(int), seed)
            else:
                model = factory(seed).fit(train[feats],
                                          train["direction_label"].astype(int))
        proba = _expand_proba(model, frame[feats].iloc[[p]])[0]
        pred = int(np.argmax(proba))
        actual = int(y_all.iloc[p])
        onehot = np.zeros(3)
        onehot[actual] = 1.0
        briers.append(float(((proba - onehot) ** 2).sum()))
        hits.append(pred == actual)
        acts.append(actual)
        preds.append(pred)
    bal = np.mean([
        np.mean([p == a for p, a in zip(preds, acts, strict=True) if a == cls])
        for cls in set(acts)
    ])
    return {"n": len(eval_pos), "accuracy": float(np.mean(hits)),
            "balanced_accuracy": float(bal), "brier": float(np.mean(briers)),
            "briers": np.array(briers),
            "dates": [frame.index[p].date().isoformat() for p in eval_pos]}


def block_ci(diff: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(diff)
    starts = n - BLOCK + 1
    means = []
    for _ in range(DRAWS):
        idx = np.concatenate([
            np.arange(s, s + BLOCK)
            for s in rng.integers(0, starts, size=int(np.ceil(n / BLOCK)))
        ])[:n]
        means.append(diff[idx].mean())
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> int:
    settings = Settings()
    closes = dash.load_daily_closes(settings)
    macro = dash.load_macro()
    cot = dash.load_cot()
    cfg = dash.FREQ_CFG[HKEY]
    frame, feats = dash.bar_frame(PAIR, closes, macro, cot, cfg,
                                  settings.dead_zone_mult)
    sf = stance_features(frame.index)
    frame = pd.concat([frame, sf], axis=1).dropna(subset=list(sf.columns))
    stance_cols = list(sf.columns)

    rows = []
    results = {}
    for name, factory, calibrated in [("logistic", make_b1, False),
                                      ("gradient_boosting", make_b2, True)]:
        for variant, cols in [("base", feats), ("with_stance", feats + stance_cols)]:
            r = walk_forward(frame, cols, factory, calibrated, settings.seed)
            results[(name, variant)] = r
            rows.append({"model": name, "variant": variant, "n": r["n"],
                         "accuracy": round(r["accuracy"], 3),
                         "balanced_accuracy": round(r["balanced_accuracy"], 3),
                         "brier": round(r["brier"], 3)})
            print(f"  {name:18s} {variant:12s} n={r['n']} "
                  f"acc={r['accuracy']:.3f} bal={r['balanced_accuracy']:.3f} "
                  f"brier={r['brier']:.3f}")
        d = results[(name, "base")]["briers"] - results[(name, "with_stance")]["briers"]
        lo, hi = block_ci(d, settings.seed)
        rows.append({"model": name, "variant": "brier_diff(base-stance)",
                     "n": len(d), "accuracy": None, "balanced_accuracy": None,
                     "brier": round(float(d.mean()), 4)})
        print(f"  {name}: paired Brier diff (base − stance) = {d.mean():+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}] "
              f"{'EXCLUDES zero' if lo > 0 or hi < 0 else 'includes zero'}")
        rows[-1]["ci"] = f"[{lo:+.4f}, {hi:+.4f}]"

    pd.DataFrame(rows).to_csv(
        ROOT / "outputs" / "case_study" / "tables" / "stance_ablation.csv",
        index=False)
    print("Wrote tables/stance_ablation.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
