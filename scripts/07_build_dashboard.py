"""07: build the prediction-dashboard payload.

Each prediction window uses input bars of its own frequency (daily /
weekly Friday-close / monthly month-end), with the same feature template
and windows scaled in bars. Per pair × frequency × model: live forecast,
recent retrained scorecard, frozen-fit context accuracy, timeline spans,
live inputs. Independent of the walk-forward study (scripts 03–06).
Output: outputs/dashboard_data.json
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
from regime_lab.models import (  # noqa: E402
    HMMDirection,
    _expand_proba,
    fit_calibrated,
    make_b1,
    make_b2,
    make_nb,
    make_rf,
)

FREQ_CFG = {
    "1d": dict(rule=None, vol=(5, 20, 60), z=252, corr=60, dz_vol=20,
               mom=(21, 63, 126, 252), hmm_states=3, trend=(50, 200),
               recent=12, context=60),
    "1w": dict(rule="W-FRI", vol=(4, 12, 26), z=52, corr=26, dz_vol=12,
               mom=(4, 13, 26, 52), hmm_states=3, trend=(10, 40),
               recent=12, context=26),
    "1m": dict(rule="ME", vol=(3, 6, 12), z=24, corr=12, dz_vol=6,
               mom=(1, 3, 6, 12), hmm_states=2, trend=(3, 12),
               recent=12, context=12),
}
MOM_NAMES = ["mom_1m", "mom_3m", "mom_6m", "mom_12m"]
FACTORIES = {"logistic": make_b1, "naive_bayes": make_nb}
CALIBRATED = {"gradient_boosting": make_b2, "random_forest": make_rf}
ENSEMBLE_OF = ["logistic", "gradient_boosting", "random_forest", "naive_bayes"]
# Rule tiers: signal column → (bucket → called class). NOTE: tsmom/trend
# windows scale per frequency, but turtle (20/10), rsi (14) and bollinger
# (20) keep their canonical FIXED bar counts — at monthly bars they become
# 20-month/14-month rules, a different animal from their daily namesakes.
# Sources:
#   tsmom  — AQR-style multi-speed time-series momentum (Moskowitz-Ooi-
#            Pedersen 2012; Hurst-Ooi-Pedersen 2017): majority sign of the
#            1m/3m/12m returns
#   turtle — the published Turtle rules, System 1 stance: enter on a
#            20-bar closing high/low, exit on a 10-bar opposite extreme
#   rsi    — the canonical retail contrarian rule: RSI(14) < 30 → call UP,
#            > 70 → call DOWN (retail fades moves; broker-flow studies)
#   boll   — Bollinger (20, 2σ) band fade, the second retail archetype
# A mapped value of None = NO RELIABLE SIGNAL: the rule is inactive (Turtle
# between breakouts, RSI mid-range, price inside the Bollinger bands). That
# is a decision state, not a market call — it is never graded, and is kept
# distinct from a FLAT forecast (an actual "move too small to matter" call).
RULE_TIERS = {
    "trend": ("trend_up", {1: 2, 0: 0}),
    #   cond_mom — volatility-gated momentum: call the momentum direction
    #   ONLY when (a) the 1-month and 12-month momentum signs agree and
    #   (b) the short-window vol is at or below its trailing median (calm
    #   half). Otherwise NO SIGNAL. With only three pairs this is not the
    #   published cross-sectional conditional currency momentum, just a
    #   time-series cousin.
    "cond_mom": ("condmom_sig", {1: 2, 0: 0, -1: None}),
    "carry": ("carry_sig", {1: 2, 0: 0}),
    "tsmom": ("tsmom_sig", {1: 2, 0: 0}),
    "turtle": ("turtle_sig", {1: 2, -1: 0, 0: None}),
    "rsi": ("rsi_sig", {0: 2, 2: 0, 1: None}),
    "bollinger": ("boll_sig", {0: 2, 2: 0, 1: None}),
    #   smart_money — follow the CFTC-reported net position of leveraged
    #   funds (real hedge-fund positioning, published weekly): call the
    #   direction the funds are positioned for. EURUSD/USDJPY only.
    "smart_money": ("smart_sig", {1: 2, 0: 0}),
}
MODEL_KEYS = ["floor", "trend", "cond_mom", "carry", "tsmom", "turtle", "rsi",
              "bollinger", "smart_money", "logistic", "gradient_boosting",
              "random_forest", "naive_bayes", "hmm", "ensemble"]
# The main view shows these six; the other tiers sit behind "view all
# experiments".
MAIN_SIX = ["floor", "cond_mom", "carry", "logistic", "gradient_boosting",
            "ensemble"]
CLASS_NAMES = ["DOWN", "FLAT", "UP"]
TARGET_DAYS = {"1d": 1, "1w": 7, "1m": 30}

# Pair orientation for carry: rate_diff = (base leg) − (quote leg), i.e. the
# yield earned by being long the pair. EURUSD is USD-per-EUR (long = long
# EUR); USDJPY/USDINR are quote-per-USD (long = long USD).
RATE_LEGS = {"EURUSD": ("rate_EZ", "rate_US"),
             "USDJPY": ("rate_US", "rate_JP"),
             "USDINR": ("rate_US", "rate_IN")}
# Publication-lag shift (days) before as-of joining. rate_US (H.15 daily)
# posts the NEXT business day, so +1; rate_EZ is a policy rate announced in
# advance, 0; the monthly OECD-style series (rate_JP, rate_IN) appear on
# FRED roughly two months late, so +60; CPI ~6 weeks. All series are
# latest-vintage (revised) values, not point-in-time (ALFRED is not wired
# in).
MACRO_LAG_DAYS = {"rate_US": 1, "rate_EZ": 0, "rate_JP": 60, "rate_IN": 60,
                  "cpi_US": 45, "cpi_EZ": 45}


def load_daily_closes(settings: Settings) -> dict[str, pd.Series]:
    raw_dir = ROOT / settings.data_root / "raw"

    def close_of(ticker: str) -> pd.Series:
        df = pd.read_parquet(raw_dir / f"{_sanitize_ticker(ticker)}.parquet")
        return df["close"].astype(float).sort_index().dropna()

    out = {pair: close_of(t) for pair, t in settings.pairs.items()}
    out["DXY"] = close_of(settings.auxiliary_tickers["DXY"])
    out["VIX"] = close_of(settings.auxiliary_tickers["VIX"])
    return out


def load_macro() -> dict[str, pd.Series]:
    macro_dir = ROOT / "data" / "raw" / "macro"
    out = {}
    for path in sorted(macro_dir.glob("*.parquet")):
        if path.stem.startswith("cot_"):
            continue  # COT files have their own loader/shape
        s = pd.read_parquet(path)["value"]
        lag = MACRO_LAG_DAYS.get(path.stem, 30)
        s.index = s.index + pd.Timedelta(days=lag)
        out[path.stem] = s
    return out


# COT smart-money orientation: EURO FX futures are long-EUR (aligned with
# EURUSD); JAPANESE YEN futures are long-yen (opposite to USDJPY). The
# parquet index is already the publication date (script 09), so as-of
# joins are causal. No rupee contract exists — USDINR has no COT tier.
COT_MAP = {"EURUSD": ("cot_EUR", 1), "USDJPY": ("cot_JPY", -1)}


def load_cot() -> dict[str, pd.DataFrame]:
    macro_dir = ROOT / "data" / "raw" / "macro"
    return {
        p.stem: pd.read_parquet(p)
        for p in sorted(macro_dir.glob("cot_*.parquet"))
    }


def _asof(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Latest-known value of ``series`` at each bar date (causal ffill)."""
    return series.reindex(series.index.union(index)).ffill().reindex(index)


def _resample(s: pd.Series, rule: str | None) -> pd.Series:
    if rule is None:
        return s
    out = s.resample(rule).last().dropna()
    # Incomplete-period guard: drop the final weekly/monthly bar unless the
    # underlying daily data reaches its period-end label, so no forecast is
    # ever graded against a bar that has not closed. (A holiday-shortened
    # final week is conservatively dropped too.)
    if len(out) and s.index[-1] < out.index[-1]:
        out = out.iloc[:-1]
    return out


def bar_frame(
    pair: str, closes: dict[str, pd.Series], macro: dict[str, pd.Series],
    cot: dict[str, pd.DataFrame], cfg: dict, dead_zone_mult: float,
) -> tuple[pd.DataFrame, list[str]]:
    """Feature + next-bar-label frame at one frequency, and its feature list."""
    pair_close, dxy, vix = closes[pair], closes["DXY"], closes["VIX"]
    idx = pair_close.index.intersection(dxy.index).intersection(vix.index)
    close = _resample(pair_close.loc[idx], cfg["rule"])
    # Yahoo stamps FX daily closes one bar EARLIER than the index closes
    # (corr(dxy_ret_t, fx_ret_{t+1}) = −0.87 vs −0.10 contemporaneous), so
    # "same-day" DXY/VIX values would sit inside the return being predicted.
    # Lag both one trading day so every DXY/VIX value is known strictly
    # before the FX bar close it joins to.
    dxy = _resample(dxy.loc[idx].shift(1), cfg["rule"])
    vix = _resample(vix.loc[idx].shift(1), cfg["rule"])

    f = pd.DataFrame(index=close.index)
    f["close"] = close
    lr = np.log(close).diff()
    f["log_return"] = lr
    for name, w in zip(["vol_s", "vol_m", "vol_l"], cfg["vol"], strict=True):
        f[name] = lr.rolling(w).std()
    f["ret_z"] = (lr - lr.rolling(cfg["z"]).mean()) / lr.rolling(cfg["z"]).std()
    f["dxy_return"] = np.log(dxy).diff()
    f["vix_level"] = vix
    f["corr_dxy"] = lr.rolling(cfg["corr"]).corr(f["dxy_return"])

    # Multi-horizon momentum + cross-pair features.
    log_close = np.log(close)
    for name, w in zip(MOM_NAMES, cfg["mom"], strict=True):
        f[name] = log_close.diff(w)
    cross_cols = []
    for other in closes:
        if other in (pair, "DXY", "VIX"):
            continue
        oc = _resample(closes[other], cfg["rule"])
        olog = np.log(oc)
        f[f"xret_{other}"] = _asof(olog.diff(), f.index)
        f[f"xmom_{other}"] = _asof(olog.diff(cfg["mom"][1]), f.index)
        cross_cols += [f"xret_{other}", f"xmom_{other}"]

    # Rate differential (all pairs) + value gap (EURUSD only; no live
    # JP/IN CPI on FRED).
    base_leg, quote_leg = RATE_LEGS[pair]
    rate_diff = _asof(macro[base_leg], f.index) - _asof(macro[quote_leg], f.index)
    f["rate_diff"] = rate_diff
    f["rate_diff_chg"] = rate_diff.diff(cfg["mom"][1])
    value_cols = []
    if pair == "EURUSD":
        rer = np.log(close) + np.log(_asof(macro["cpi_EZ"], f.index)) \
            - np.log(_asof(macro["cpi_US"], f.index))
        f["value_gap"] = rer - rer.rolling(cfg["z"] * 2, min_periods=cfg["z"]).mean()
        value_cols = ["value_gap"]

    fast, slow = cfg["trend"]
    f["trend_up"] = (close.rolling(fast).mean() > close.rolling(slow).mean()).astype(int)
    f["carry_sig"] = (rate_diff > 0).astype(int)

    # Institutional signals: multi-speed momentum votes and the Turtle
    # Donchian stance machine (20-bar entry, 10-bar exit).
    votes = (np.sign(f["mom_1m"]) + np.sign(f["mom_3m"]) + np.sign(f["mom_12m"]))
    f["tsmom_sig"] = np.where(votes > 0, 1, np.where(votes < 0, 0,
                              (f["mom_12m"] >= 0).astype(int)))

    # Conditional momentum: momentum call only when fast/slow momentum
    # agree AND vol is in its calm half (trailing median of the short vol
    # window — strictly causal).
    vol_med = f["vol_s"].rolling(cfg["z"], min_periods=cfg["dz_vol"]).median()
    agree_up = (f["mom_1m"] > 0) & (f["mom_12m"] > 0)
    agree_dn = (f["mom_1m"] < 0) & (f["mom_12m"] < 0)
    calm = f["vol_s"] <= vol_med
    f["condmom_sig"] = np.where(calm & agree_up, 1,
                                np.where(calm & agree_dn, 0, -1))
    hi20 = close.shift(1).rolling(20).max()
    lo20 = close.shift(1).rolling(20).min()
    hi10 = close.shift(1).rolling(10).max()
    lo10 = close.shift(1).rolling(10).min()
    stance, cur = [], 0
    for c, h20, l20, h10, l10 in zip(close, hi20, lo20, hi10, lo10, strict=True):
        if not np.isnan(h20):
            if c > h20:
                cur = 1
            elif c < l20:
                cur = -1
            elif cur == 1 and c < l10:
                cur = 0
            elif cur == -1 and c > h10:
                cur = 0
        stance.append(cur)
    f["turtle_sig"] = stance

    # Retail signals: Wilder RSI(14) buckets and Bollinger (20, 2σ) buckets.
    delta = close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + avg_gain / avg_loss.replace(0, np.nan))
    f["rsi_14"] = rsi
    f["rsi_sig"] = np.where(rsi < 30, 0, np.where(rsi > 70, 2, 1))
    sma20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std()
    f["boll_sig"] = np.where(close < sma20 - 2 * sd20, 0,
                             np.where(close > sma20 + 2 * sd20, 2, 1))

    # Smart-money (COT) features + signal, where a contract exists.
    cot_cols = []
    if pair in COT_MAP:
        code, sign = COT_MAP[pair]
        lev = sign * cot[code]["lev_net"]
        lev_z = (lev - lev.rolling(104, min_periods=52).mean()) / lev.rolling(
            104, min_periods=52
        ).std()
        f["smart_pos_z"] = _asof(lev_z, f.index)
        f["smart_chg"] = _asof(lev.diff(4), f.index) / 1000.0
        f["smart_sig"] = (_asof(lev, f.index) > 0).astype(int)
        cot_cols = ["smart_pos_z", "smart_chg"]

    # Next-bar direction label with a dead zone scaled to trailing bar vol.
    fwd = log_close.shift(-1) - log_close
    thr = dead_zone_mult * lr.rolling(cfg["dz_vol"]).std()
    f["threshold_k"] = thr
    label = pd.Series(np.nan, index=f.index)
    valid = fwd.notna() & thr.notna()
    label[valid & (fwd.abs() <= thr)] = 1.0
    label[valid & (fwd > thr)] = 2.0
    label[valid & (fwd < -thr)] = 0.0
    f["direction_label"] = label

    # Turbulence target: is the next bar stormier than typical (trailing
    # median of |return|)? Roughly a 50/50 split by construction. The
    # magnitude columns feed the side-analysis chart.
    typical = lr.abs().rolling(cfg["z"], min_periods=cfg["dz_vol"]).median()
    f["turb_typical"] = typical
    f["next_abs_move"] = fwd.abs()
    f["turb_label"] = (fwd.abs() > typical).astype(float).where(valid)

    feats = (["log_return", "vol_s", "vol_m", "vol_l", "ret_z", "dxy_return",
              "vix_level", "corr_dxy"] + MOM_NAMES + cross_cols
             + ["rate_diff", "rate_diff_chg"] + value_cols + cot_cols)
    signal_cols = [col for col, _ in RULE_TIERS.values() if col in f.columns]
    return f.dropna(subset=feats + ["threshold_k", "rsi_14"] + signal_cols), feats


def _rule_freqs(y: pd.Series, signal: pd.Series, buckets) -> dict[int, np.ndarray]:
    freqs = {}
    for sig in buckets:
        counts = np.bincount(y[signal == sig], minlength=3) + 1.0  # Laplace
        freqs[sig] = counts / counts.sum()
    return freqs


def fit_model(key, frame, feats, cut_pos, cfg, seed, cache):
    """Fit using only rows whose labels were resolved at the prediction
    date (positions ≤ cut_pos). Cached per (key, cut_pos) so the ensemble
    reuses its members' fits."""
    if (key, cut_pos) in cache:
        return cache[(key, cut_pos)]
    train = frame.iloc[: cut_pos + 1]
    train = train[train["direction_label"].notna()]
    y = train["direction_label"].astype(int)

    if key == "floor":
        model = np.bincount(y, minlength=3) / len(y)
    elif key in RULE_TIERS:
        col, mapping = RULE_TIERS[key]
        model = _rule_freqs(y, train[col], mapping.keys())
    elif key == "hmm":
        model = HMMDirection(n_states=cfg["hmm_states"], seed=seed).fit(
            train["log_return"], y
        )
    elif key in CALIBRATED:
        model = fit_calibrated(CALIBRATED[key], train[feats], y, seed)
    elif key == "ensemble":
        model = [fit_model(k, frame, feats, cut_pos, cfg, seed, cache)
                 for k in ENSEMBLE_OF]
    else:
        model = FACTORIES[key](seed).fit(train[feats], y)
    cache[(key, cut_pos)] = model
    return model


def predict_at(key, model, frame, feats, pos):
    row = frame.iloc[[pos]]
    if key == "floor":
        r = float(row["log_return"].iloc[0])
        t = float(row["threshold_k"].iloc[0])
        return (2 if r > t else 0 if r < -t else 1), model
    if key in RULE_TIERS:
        col, mapping = RULE_TIERS[key]
        sig = int(row[col].iloc[0])
        return mapping[sig], model[sig]
    if key == "hmm":
        proba = model.predict_proba_at_end(frame["log_return"].iloc[: pos + 1])
        return int(proba.argmax()), proba
    if key == "ensemble":
        proba = np.mean([_expand_proba(m, row[feats]) for m in model], axis=0)[0]
        return int(proba.argmax()), proba
    proba = _expand_proba(model, row[feats])[0]
    return int(proba.argmax()), proba


def turbulence_summary(frame, feats_vol, ctx_pos, live_pos, seed):
    """One frozen logistic fit predicting next-bar turbulence."""
    cut = ctx_pos[0] - 1
    train = frame.iloc[: cut + 1]
    train = train[train["turb_label"].notna()]
    model = make_b1(seed).fit(train[feats_vol], train["turb_label"].astype(int))
    preds = model.predict(frame[feats_vol].iloc[ctx_pos])
    actual = frame["turb_label"].iloc[ctx_pos].astype(int).to_numpy()
    p_live = float(model.predict_proba(frame[feats_vol].iloc[[live_pos]])[0][-1])
    base = float(train["turb_label"].mean())
    return {
        "live_p_turbulent": round(p_live, 3),
        "context": {"n": len(ctx_pos), "hits": int((preds == actual).sum()),
                    "accuracy": round(float((preds == actual).mean()), 3)},
        "base_rate": round(base, 3),
        # Per-bar series for the side-analysis chart: what actually
        # happened next vs the "usual" line, and whether the call landed.
        "series": [
            {
                "date": frame.index[p].date().isoformat(),
                "next_abs_move": round(float(frame["next_abs_move"].iloc[p]), 5),
                "typical": round(float(frame["turb_typical"].iloc[p]), 5),
                "pred_stormy": int(preds[i]),
                "hit": bool(preds[i] == actual[i]),
            }
            for i, p in enumerate(ctx_pos)
        ],
    }


def build_pair(pair, closes, macro, cot, settings):
    out: dict = {"horizons": {}}
    for hkey, cfg in FREQ_CFG.items():
        frame, feats = bar_frame(pair, closes, macro, cot, cfg, settings.dead_zone_mult)
        pair_keys = [
            k for k in MODEL_KEYS
            if k not in RULE_TIERS or RULE_TIERS[k][0] in frame.columns
        ]
        resolved = frame["direction_label"].notna().to_numpy()
        last_resolved = int(np.where(resolved)[0].max())
        live_pos = len(frame) - 1

        recent_pos = [last_resolved - i for i in range(cfg["recent"])][::-1]
        ctx_pos = [last_resolved - cfg["recent"] - j for j in range(cfg["context"])][::-1]
        cache: dict = {}

        # Majority-class baseline of each evaluation window (persistence is
        # already the "floor" model row; 33% guessing is retired as the sole
        # baseline).
        ctx_actuals = [int(frame["direction_label"].iloc[p]) for p in ctx_pos]
        rec_actuals = [int(frame["direction_label"].iloc[p]) for p in recent_pos]
        ctx_counts = np.bincount(ctx_actuals, minlength=3)
        rec_counts = np.bincount(rec_actuals, minlength=3)
        baselines = {
            "context_majority": {"class": CLASS_NAMES[int(ctx_counts.argmax())],
                                 "hits": int(ctx_counts.max()),
                                 "n": len(ctx_actuals)},
            "recent_majority": {"class": CLASS_NAMES[int(rec_counts.argmax())],
                                "hits": int(rec_counts.max()),
                                "n": len(rec_actuals)},
        }

        def _name(pred):  # None = inactive rule → decision state, not a call
            return "NO SIGNAL" if pred is None else CLASS_NAMES[pred]

        def _bal_acc(acts, preds):
            per_class = [
                sum(p == cls for a, p in zip(acts, preds, strict=True) if a == cls)
                / sum(a == cls for a in acts)
                for cls in set(acts)
            ]
            return sum(per_class) / len(per_class)

        models_out: dict = {}
        for key in pair_keys:
            ctx_model = fit_model(key, frame, feats, ctx_pos[0] - 1, cfg,
                                  settings.seed, cache)
            ctx_hits = ctx_active = 0
            acts_a, preds_a, briers, halves = [], [], [], [0, 0]
            for j, p in enumerate(ctx_pos):
                pred, proba = predict_at(key, ctx_model, frame, feats, p)
                if pred is None:
                    continue  # NO SIGNAL bars are not graded
                actual = int(frame["direction_label"].iloc[p])
                hit = pred == actual
                ctx_active += 1
                ctx_hits += hit
                halves[j >= len(ctx_pos) // 2] += hit
                acts_a.append(actual)
                preds_a.append(pred)
                onehot = np.zeros(3)
                onehot[actual] = 1.0
                briers.append(float(((np.asarray(proba, float) - onehot) ** 2).sum()))
            recent = []
            for p in recent_pos:
                model = fit_model(key, frame, feats, p - 1, cfg, settings.seed, cache)
                pred, _ = predict_at(key, model, frame, feats, p)
                actual = int(frame["direction_label"].iloc[p])
                recent.append({"date": frame.index[p].date().isoformat(),
                               "pred": _name(pred),
                               "actual": CLASS_NAMES[actual],
                               "hit": None if pred is None else pred == actual})
            live_model = fit_model(key, frame, feats, live_pos - 1, cfg,
                                   settings.seed, cache)
            live_pred, live_proba = predict_at(key, live_model, frame, feats, live_pos)
            models_out[key] = {
                "live": {"pred": _name(live_pred),
                         "proba": [round(float(p), 3) for p in live_proba]},
                "recent": recent,
                "context": {"n": cfg["context"], "active": int(ctx_active),
                            "hits": int(ctx_hits),
                            "accuracy": round(ctx_hits / ctx_active, 3)
                            if ctx_active else None,
                            "balanced_accuracy": round(_bal_acc(acts_a, preds_a), 3)
                            if ctx_active else None,
                            "brier": round(float(np.mean(briers)), 3)
                            if ctx_active else None,
                            "half_hits": [int(h) for h in halves]},
                # True/False for the isotonic-calibrated tiers (False = the
                # calibration slice was too small, probabilities are raw);
                # None for tiers where calibration does not apply.
                "calibrated": getattr(live_model, "was_calibrated_", None)
                if key in CALIBRATED else None,
            }

        # Fair comparison: floor and plain multi-speed momentum re-scored
        # on exactly the bars where conditional momentum was active, so the
        # test observations are identical.
        if "cond_mom" in models_out:
            cm_model = fit_model("cond_mom", frame, feats, ctx_pos[0] - 1,
                                 cfg, settings.seed, cache)
            active_pos = [p for p in ctx_pos
                          if predict_at("cond_mom", cm_model, frame, feats, p)[0]
                          is not None]
            on_active = {"n": len(active_pos)}
            for k in ("floor", "tsmom"):
                mdl = fit_model(k, frame, feats, ctx_pos[0] - 1, cfg,
                                settings.seed, cache)
                on_active[k] = int(sum(
                    predict_at(k, mdl, frame, feats, p)[0]
                    == int(frame["direction_label"].iloc[p])
                    for p in active_pos
                ))
            models_out["cond_mom"]["on_active"] = on_active

        as_of = frame.index[live_pos]
        out["horizons"][hkey] = {
            "models": models_out,
            "baselines": baselines,
            "turbulence": turbulence_summary(
                frame, ["vol_s", "vol_m", "vol_l", "log_return"],
                ctx_pos, live_pos, settings.seed,
            ),
            "bars": {
                "dates": [d.date().isoformat() for d in frame.index],
                "close": [round(float(v), 5) for v in frame["close"]],
            },
            "timeline": {
                "train_start": frame.index[0].date().isoformat(),
                "train_end": frame.index[recent_pos[0] - 1].date().isoformat(),
                "recent_start": frame.index[recent_pos[0]].date().isoformat(),
                "as_of": as_of.date().isoformat(),
                "target_end": (as_of + pd.Timedelta(days=TARGET_DAYS[hkey]))
                .date().isoformat(),
            },
            "inputs": {
                name: round(float(frame[name].iloc[live_pos]), 5)
                for name in feats
            } | {
                "trend_up": int(frame["trend_up"].iloc[live_pos]),
                "rsi_14": round(float(frame["rsi_14"].iloc[live_pos]), 1),
                "turtle_sig": int(frame["turtle_sig"].iloc[live_pos]),
                "tsmom_sig": int(frame["tsmom_sig"].iloc[live_pos]),
            },
        }
        print(f"  {pair} {hkey}: {len(frame)} bars, {len(feats)} features, done")

    daily = closes[pair].iloc[-2700:]
    out["history"] = {
        "dates": [d.date().isoformat() for d in daily.index],
        "close": [round(float(v), 5) for v in daily],
    }
    return out


def main() -> int:
    settings = Settings()
    closes = load_daily_closes(settings)
    macro = load_macro()
    cot = load_cot()
    payload = {"as_of": None, "main_six": MAIN_SIX, "pairs": {}}
    for pair in settings.pairs:
        payload["pairs"][pair] = build_pair(pair, closes, macro, cot, settings)
        payload["as_of"] = payload["pairs"][pair]["horizons"]["1d"]["timeline"]["as_of"]

    out_path = ROOT / "outputs" / "dashboard_data.json"
    out_path.write_text(json.dumps(payload, allow_nan=False))
    print(f"\nWrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB), "
          f"as of {payload['as_of']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
