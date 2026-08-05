"""17: calibration curve for EURUSD direction on the held-out block.

Re-scores logistic (B1), gradient boosting (B2) and a 3-state HMM on the
Block-6 confirmation rows (n=322), saves per-prediction P(up) vs outcome
to tables/calibration_eurusd.csv, prints a 5-bin reliability table, and
writes figures/calibration_eurusd.svg as hand-built markup (no plotting
library). actual_up = 1 only for UP; FLAT and DOWN both count as 0, since
the chart calibrates the stated "up" probability.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime_lab.config import Settings  # noqa: E402
from regime_lab.data.splits import split_by_block  # noqa: E402
from regime_lab.dataset import TARGET_COLUMN, load_pair_dataset  # noqa: E402
from regime_lab.models import (  # noqa: E402
    MODEL_FEATURES,
    HMMDirection,
    _expand_proba,
    make_b1,
    make_b2,
)

COLORS = {"logistic": "#2a78d6", "gradient_boosting": "#eb6834", "hmm": "#1baf7a"}
NAMES = {"logistic": "logistic regression", "gradient_boosting": "gradient boosting",
         "hmm": "hidden Markov model"}
BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
MIN_BIN = 10

PLOT0, PLOT1 = 90, 610  # plot-area corners in a 700x700 viewBox
SPAN = PLOT1 - PLOT0


def collect() -> pd.DataFrame:
    settings = Settings()
    df = load_pair_dataset("EURUSD", settings, ROOT)
    block = next(b for b in settings.walk_forward_blocks if b.block_id == 6)
    train, test = split_by_block(df, block, purge=settings.direction_k)
    y_train = train[TARGET_COLUMN]
    actual_up = (test[TARGET_COLUMN] == 2).astype(int).to_numpy()

    rows = []
    for key, factory in [("logistic", make_b1), ("gradient_boosting", make_b2)]:
        model = factory(settings.seed).fit(train[MODEL_FEATURES], y_train)
        p_up = _expand_proba(model, test[MODEL_FEATURES])[:, 2]
        rows += [{"date": d.date().isoformat(), "model": key,
                  "prob_up": round(float(p), 4), "actual_up": int(a)}
                 for d, p, a in zip(test.index, p_up, actual_up, strict=True)]

    # HMM: fit on the training returns, then filter causally up to each
    # test date so no test-period information leaks into the state estimate.
    hmm = HMMDirection(n_states=3, seed=settings.seed).fit(
        train["log_return"], y_train)
    returns = df["log_return"]
    for d, a in zip(test.index, actual_up, strict=True):
        pos = returns.index.get_loc(d)
        p = hmm.predict_proba_at_end(returns.iloc[: pos + 1])[2]
        rows.append({"date": d.date().isoformat(), "model": "hmm",
                     "prob_up": round(float(p), 4), "actual_up": int(a)})
    return pd.DataFrame(rows)


def bin_table(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key in COLORS:
        g = preds[preds["model"] == key]
        for lo, hi in BINS:
            m = (g["prob_up"] >= lo) & (g["prob_up"] < (hi if hi < 1 else 1.01))
            sel = g[m]
            rows.append({"model": key, "bin": f"[{lo:.1f}-{hi:.1f})",
                         "n": len(sel),
                         "mean_prob": round(float(sel["prob_up"].mean()), 4)
                         if len(sel) else None,
                         "frac_up": round(float(sel["actual_up"].mean()), 4)
                         if len(sel) else None,
                         "plotted": len(sel) >= MIN_BIN})
    return pd.DataFrame(rows)


def x_px(v: float) -> float:
    return PLOT0 + v * SPAN


def y_px(v: float) -> float:
    return PLOT1 - v * SPAN


def marker(key: str, x: float, y: float, r: float) -> str:
    c = COLORS[key]
    if key == "logistic":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{c}"/>'
    if key == "gradient_boosting":
        s = r * 0.89  # half-side, roughly area-matched to the circle
        return (f'<rect x="{x - s:.1f}" y="{y - s:.1f}" width="{2 * s:.1f}" '
                f'height="{2 * s:.1f}" fill="{c}"/>')
    pts = " ".join(f"{x + r * np.sin(t):.1f},{y - r * np.cos(t):.1f}"
                   for t in (0, 2.0944, 4.1888))
    return f'<polygon points="{pts}" fill="{c}"/>'


def build_svg(bins: pd.DataFrame, n_held_out: int) -> str:
    plotted = bins[bins["plotted"]]
    n_lo, n_hi = plotted["n"].min(), plotted["n"].max()

    def radius(n: int) -> float:
        if n_hi == n_lo:
            return 8.5
        return 5 + 7 * (n - n_lo) / (n_hi - n_lo)

    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 700 700">',
             '<g font-family="system-ui, sans-serif">']

    # diagonal reference
    parts.append(f'<line x1="{PLOT0}" y1="{PLOT1}" x2="{PLOT1}" y2="{PLOT0}" '
                 'stroke="#C3C2B7" stroke-width="1.5" stroke-dasharray="6 6"/>')
    parts.append('<text x="530" y="152" font-size="12" fill="#C3C2B7" '
                 'transform="rotate(-45 530 152)">perfect calibration</text>')

    # ticks and labels
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        x, y = x_px(v), y_px(v)
        lab = f"{v:g}"
        parts.append(f'<line x1="{x:.1f}" y1="{PLOT1}" x2="{x:.1f}" '
                     f'y2="{PLOT1 + 6}" stroke="#898781"/>')
        parts.append(f'<text x="{x:.1f}" y="{PLOT1 + 24}" font-size="13" '
                     f'fill="#898781" text-anchor="middle">{lab}</text>')
        parts.append(f'<line x1="{PLOT0 - 6}" y1="{y:.1f}" x2="{PLOT0}" '
                     f'y2="{y:.1f}" stroke="#898781"/>')
        parts.append(f'<text x="{PLOT0 - 12}" y="{y + 4:.1f}" font-size="13" '
                     f'fill="#898781" text-anchor="end">{lab}</text>')

    # axis titles + footnote
    parts.append('<text x="350" y="655" font-size="14" fill="#52514E" '
                 'text-anchor="middle">predicted probability of up</text>')
    parts.append('<text x="30" y="350" font-size="14" fill="#52514E" '
                 'text-anchor="middle" transform="rotate(-90 30 350)">'
                 'actual frequency of up</text>')
    parts.append(f'<text x="350" y="678" font-size="12" fill="#898781" '
                 f'text-anchor="middle">EURUSD direction · {n_held_out} '
                 'held-out predictions · 5 equal-width bins</text>')

    # one polyline + sized markers per model
    for key in COLORS:
        g = plotted[plotted["model"] == key]
        pts = " ".join(f"{x_px(r.mean_prob):.1f},{y_px(r.frac_up):.1f}"
                       for r in g.itertuples())
        parts.append(f'<polyline points="{pts}" fill="none" '
                     f'stroke="{COLORS[key]}" stroke-width="2" '
                     'stroke-linejoin="round"/>')
        for r in g.itertuples():
            parts.append(marker(key, x_px(r.mean_prob), y_px(r.frac_up),
                                radius(r.n)))

    # legend in the empty corner below the diagonal
    for i, key in enumerate(COLORS):
        y = 505 + i * 30
        parts.append(marker(key, 430, y, 6))
        parts.append(f'<text x="445" y="{y + 5}" font-size="14" '
                     f'fill="#52514E">{NAMES[key]}</text>')

    parts += ["</g>", "</svg>"]
    return "\n".join(parts)


def main() -> int:
    preds = collect()
    preds.to_csv(ROOT / "outputs" / "case_study" / "tables"
                 / "calibration_eurusd.csv", index=False)
    bins = bin_table(preds)
    print(bins.to_string(index=False))
    n = preds["model"].value_counts().iloc[0]
    svg = build_svg(bins, int(n))
    out = ROOT / "outputs" / "case_study" / "figures" / "calibration_eurusd.svg"
    out.write_text(svg)
    print(f"\nWrote {out} and tables/calibration_eurusd.csv (n={n} per model)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
