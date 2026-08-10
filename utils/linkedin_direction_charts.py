"""One-off social asset — four LinkedIn-format SVGs of the EURUSD
direction results (returns-by-prediction dots, confusion grids,
cumulative-edge lines, hit/miss raster).

Not part of the case-study pipeline; the study cites the notebook and
script outputs directly.

Re-scores B1 and B2 on the Block-6 rows (post-fix data), keeps the better
one, and reduces everything to binary up/down: model_pred = P(up) > P(down),
actual = sign of the realised 5-day-ahead return, benchmark = sign of the
trailing 5-day return (persistence). Saves tables/eurusd_predictions.csv
and four hand-built 1080x1080 SVGs under figures/.
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
from regime_lab.models import MODEL_FEATURES, _expand_proba, make_b1, make_b2  # noqa: E402

UP, DOWN, REF = "#2a78d6", "#eb6834", "#C3C2B7"
INK, INK2, MUTED = "#0B0B0B", "#52514E", "#898781"
FONT = 'font-family="system-ui, sans-serif"'
FIG_DIR = ROOT / "outputs" / "case_study" / "figures"


def load_predictions() -> tuple[pd.DataFrame, str]:
    settings = Settings()
    df = load_pair_dataset("EURUSD", settings, ROOT)
    block = next(b for b in settings.walk_forward_blocks if b.block_id == 6)
    train, test = split_by_block(df, block, purge=settings.direction_k)
    y_train = train[TARGET_COLUMN]

    log_close = np.log(df["close"].astype(float))
    fwd = (log_close.shift(-settings.direction_k) - log_close) * 100
    fwd_ret = fwd.loc[test.index]
    if fwd_ret.isna().any():
        raise SystemExit("actual_return: forward closes missing for some test rows")
    actual = (fwd_ret > 0).astype(int)
    benchmark = (test["trailing_ret_5d"] > 0).astype(int)

    cand = {}
    for key, factory in [("logistic", make_b1), ("gradient_boosting", make_b2)]:
        model = factory(settings.seed).fit(train[MODEL_FEATURES], y_train)
        proba = _expand_proba(model, test[MODEL_FEATURES])
        pred = (proba[:, 2] > proba[:, 0]).astype(int)
        cand[key] = pred
        print(f"{key}: accuracy {np.mean(pred == actual):.3f}")
    print(f"benchmark (persistence): accuracy {np.mean(benchmark == actual):.3f}")
    best = max(cand, key=lambda k: np.mean(cand[k] == actual))
    print(f"best model: {best}")

    out = pd.DataFrame({
        "date": [d.date().isoformat() for d in test.index],
        "model_pred": cand[best],
        "benchmark_pred": benchmark.to_numpy(),
        "actual": actual.to_numpy(),
        "actual_return": fwd_ret.round(4).to_numpy(),
    })
    out.to_csv(ROOT / "outputs" / "case_study" / "tables"
               / "eurusd_predictions.csv", index=False)
    return out, best


def svg_open() -> list[str]:
    return ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 1080">',
            f"<g {FONT}>"]


def footnote(n: int) -> str:
    return (f'<text x="540" y="1050" font-size="22" fill="{MUTED}" '
            f'text-anchor="middle">EURUSD direction · {n} held-out '
            "predictions</text>")


def chart_returns(df: pd.DataFrame, seed: int) -> str:
    rng = np.random.default_rng(seed)
    top, bot = 80, 900
    rmax = float(df["actual_return"].abs().max()) * 1.08
    y = lambda v: (top + bot) / 2 - v / rmax * (bot - top) / 2  # noqa: E731

    parts = svg_open()
    parts.append(f'<line x1="120" y1="{y(0):.0f}" x2="960" y2="{y(0):.0f}" '
                 f'stroke="{INK2}" stroke-width="2"/>')
    means = {}
    for pred, cx, label in [(1, 330, "days it said up"),
                            (0, 750, "days it said down")]:
        g = df[df["model_pred"] == pred]
        color = UP if pred else DOWN
        for v in g["actual_return"]:
            x = cx + rng.uniform(-90, 90)
            parts.append(f'<circle cx="{x:.1f}" cy="{y(v):.1f}" r="5" '
                         f'fill="{color}" opacity="0.45"/>')
        m = float(g["actual_return"].mean())
        means[label] = (m, len(g))
        parts.append(f'<line x1="{cx - 100}" y1="{y(m):.1f}" x2="{cx + 100}" '
                     f'y2="{y(m):.1f}" stroke="{INK}" stroke-width="5"/>')
        parts.append(f'<text x="{cx}" y="955" font-size="28" fill="{INK2}" '
                     f'text-anchor="middle">{label}</text>')
    parts.append(f'<text x="985" y="{y(0) + 8:.0f}" font-size="22" '
                 f'fill="{MUTED}">0%</text>')
    parts += [footnote(len(df)), "</g>", "</svg>"]
    for label, (m, n) in means.items():
        print(f"  {label}: mean return {m:+.3f}% (n={n})")
    return "\n".join(parts)


def chart_confusion(df: pd.DataFrame) -> str:
    def grid(pred_col):
        c = np.zeros((2, 2), dtype=int)  # rows: pred up/down; cols: act up/down
        for pr in (1, 0):
            for ac in (1, 0):
                c[1 - pr, 1 - ac] = int(((df[pred_col] == pr)
                                         & (df["actual"] == ac)).sum())
        return c

    grids = [("the model", grid("model_pred"), 250),
             ("the lazy benchmark", grid("benchmark_pred"), 660)]
    n_lo = min(g.min() for _, g, _ in grids)
    n_hi = max(g.max() for _, g, _ in grids)
    cell, top = 190, 320
    parts = svg_open()
    # one shared set of row labels, left of the first grid
    for i, lab in enumerate(["said up", "said down"]):
        parts.append(f'<text x="240" y="{top + cell * i + cell / 2:.0f}" '
                     f'font-size="24" fill="{INK2}" '
                     f'text-anchor="end">{lab}</text>')
    for title, c, x0 in grids:
        parts.append(f'<text x="{x0 + cell}" y="250" font-size="34" '
                     f'fill="{INK}" text-anchor="middle">{title}</text>')
        for j, lab in enumerate(["actually up", "actually down"]):
            parts.append(f'<text x="{x0 + cell * j + cell / 2:.0f}" y="298" '
                         f'font-size="24" fill="{INK2}" '
                         f'text-anchor="middle">{lab}</text>')
        for i in range(2):
            row_total = c[i].sum()
            for j in range(2):
                x, yv = x0 + cell * j, top + cell * i
                op = 0.1 + 0.5 * (c[i, j] - n_lo) / max(n_hi - n_lo, 1)
                border = (f' stroke="{INK}" stroke-width="3"' if i == j else "")
                parts.append(f'<rect x="{x}" y="{yv}" width="{cell}" '
                             f'height="{cell}" fill="{UP}" '
                             f'opacity="{op:.2f}"{border}/>')
                parts.append(f'<text x="{x + cell / 2:.0f}" '
                             f'y="{yv + cell / 2 + 5:.0f}" font-size="60" '
                             f'fill="{INK}" text-anchor="middle">{c[i, j]}</text>')
                pct = 100 * c[i, j] / row_total if row_total else 0
                parts.append(f'<text x="{x + cell / 2:.0f}" '
                             f'y="{yv + cell / 2 + 45:.0f}" font-size="26" '
                             f'fill="{INK2}" text-anchor="middle">'
                             f'{pct:.0f}% of row</text>')
        print(f"  {title}: {c.tolist()}")
    parts += [footnote(len(df)), "</g>", "</svg>"]
    return "\n".join(parts)


def chart_cumulative(df: pd.DataFrame) -> str:
    n = len(df)
    cum = {"model": np.cumsum(np.where(df["model_pred"] == df["actual"], 1, -1)),
           "benchmark": np.cumsum(np.where(df["benchmark_pred"] == df["actual"],
                                           1, -1))}
    lo = min(0, min(c.min() for c in cum.values()))
    hi = max(0, max(c.max() for c in cum.values()))
    lo, hi = 10 * np.floor(lo / 10), 10 * np.ceil(hi / 10)
    x0, x1, y0, y1 = 140, 900, 120, 920
    x = lambda i: x0 + i / (n - 1) * (x1 - x0)  # noqa: E731
    y = lambda v: y1 - (v - lo) / (hi - lo) * (y1 - y0)  # noqa: E731

    parts = svg_open()
    parts.append(f'<line x1="{x0}" y1="{y(0):.0f}" x2="{x1}" y2="{y(0):.0f}" '
                 f'stroke="{INK2}" stroke-dasharray="8 8"/>')
    for v in np.arange(lo, hi + 1, 10):
        parts.append(f'<text x="{x0 - 14}" y="{y(v) + 8:.0f}" font-size="22" '
                     f'fill="{MUTED}" text-anchor="end">{v:.0f}</text>')
    for i, lab in [(0, "0"), (n - 1, str(n))]:
        parts.append(f'<text x="{x(i):.0f}" y="{y1 + 40}" font-size="22" '
                     f'fill="{MUTED}" text-anchor="middle">{lab}</text>')
    parts.append(f'<text x="{(x0 + x1) / 2:.0f}" y="{y1 + 80}" font-size="24" '
                 f'fill="{INK2}" text-anchor="middle">prediction number</text>')
    for key, color, width in [("benchmark", REF, 3), ("model", UP, 4)]:
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(cum[key]))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                     f'stroke-width="{width}" stroke-linejoin="round"/>')
        parts.append(f'<text x="{x1 + 12}" y="{y(cum[key][-1]) + 8:.0f}" '
                     f'font-size="26" fill="{color if key == "model" else INK2}">'
                     f"{key}</text>")
        print(f"  {key}: final net correct {int(cum[key][-1]):+d}")
    parts += [footnote(n), "</g>", "</svg>"]
    return "\n".join(parts)


def chart_raster(df: pd.DataFrame) -> str:
    n = len(df)
    # 840 not the spec's 900: left labels + 40px percentages don't fit a
    # 900px band inside the 1080 frame
    bw, bh, gap, x0 = 840, 220, 60, 150
    top = (1080 - 2 * bh - gap) / 2 - 30
    w = bw / n
    parts = svg_open()
    for row, (key, col) in enumerate([("model", "model_pred"),
                                      ("benchmark", "benchmark_pred")]):
        yv = top + row * (bh + gap)
        hits = (df[col] == df["actual"]).to_numpy()
        for i, hit in enumerate(hits):
            parts.append(f'<rect x="{x0 + i * w:.2f}" y="{yv:.0f}" '
                         f'width="{w:.2f}" height="{bh}" '
                         f'fill="{UP if hit else DOWN}"/>')
        acc = 100 * hits.mean()
        parts.append(f'<text x="{x0 - 12}" y="{yv + bh / 2 + 9:.0f}" '
                     f'font-size="28" fill="{INK2}" '
                     f'text-anchor="end">{key}</text>')
        parts.append(f'<text x="{x0 + bw + 14}" y="{yv + bh / 2 + 13:.0f}" '
                     f'font-size="40" fill="{INK}">{acc:.0f}%</text>')
        print(f"  {key}: raster accuracy {acc:.1f}%")
    parts += [footnote(n), "</g>", "</svg>"]
    return "\n".join(parts)


def main() -> int:
    settings = Settings()
    df, best = load_predictions()
    charts = {
        "returns_by_prediction.svg": chart_returns(df, settings.seed),
        "confusion_side_by_side.svg": chart_confusion(df),
        "cumulative_edge.svg": chart_cumulative(df),
        "prediction_raster.svg": chart_raster(df),
    }
    for name, svg in charts.items():
        (FIG_DIR / name).write_text(svg)
        print(f"wrote {FIG_DIR / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
