"""One-off social asset — past_help.png, a 1080x1080 summary card:
size memory vs direction memory (ten-lag means), the "5-8x" callout.

Not part of the case-study pipeline; the values are recomputed from
data/raw on every run, never hardcoded."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PAIRS = {"EURUSD": "EURUSD_X", "USDJPY": "JPY_X", "USDINR": "INR_X"}
GREY, BLUE, INK, MUTED = "#B5B4AC", "#14497F", "#0B0B0B", "#898781"

closes = {n: pd.read_parquet(RAW / f"{f}.parquet")["close"].dropna()
          for n, f in (PAIRS | {"DXY": "DX-Y_NYB", "VIX": "_VIX"}).items()}
lagged = {"DXY": closes["DXY"].shift(1), "VIX": closes["VIX"].shift(1)}


def frame_returns(pair):  # same row set as the published figure
    px = closes[pair]
    common = px.index.intersection(closes["DXY"].index).intersection(closes["VIX"].index)
    px = px.loc[common]
    r = np.log(px).diff()
    f = pd.DataFrame(index=common)
    f["log_return"] = r
    for w in (5, 20, 60):
        f[f"vol_{w}d"] = r.rolling(w).std()
    f["ret_z"] = (r - r.rolling(252).mean()) / r.rolling(252).std()
    f["dxy_return"] = np.log(lagged["DXY"].reindex(common)).diff()
    f["vix"] = lagged["VIX"].reindex(common)
    f["corr_dxy"] = r.rolling(60).corr(f["dxy_return"])
    fwd = np.log(px).shift(-5) - np.log(px)
    dead = 0.25 * f["vol_20d"] * np.sqrt(5)
    f["label"] = np.select([fwd > dead, fwd < -dead], [2, 0], default=1).astype(float)
    f.loc[fwd.isna() | dead.isna(), "label"] = np.nan
    return f.dropna()["log_return"]


means = {}
for p in PAIRS:
    r = frame_returns(p)
    means[p] = (np.mean([abs(r.autocorr(k)) for k in range(1, 11)]),
                np.mean([r.abs().autocorr(k) for k in range(1, 11)]))
for p, (w, b) in means.items():
    print(f"{p}: direction {w:.3f}  size {b:.3f}  ratio {b / w:.1f}x")

plt.rcParams["font.family"] = "DejaVu Sans"
PX = 72 / 100
fig = plt.figure(figsize=(10.8, 10.8), dpi=100, facecolor="white")

fig.text(0.06, 0.945, "The market hides which way it will go.",
         fontsize=40 * PX, color=INK, weight="medium", va="top")
fig.text(0.06, 0.895, "It broadcasts how rough the ride will be.",
         fontsize=40 * PX, color=INK, weight="medium", va="top")

# --- bars ---
ax = fig.add_axes([0.21, 0.3333, 0.50, 0.4120])
xmax = max(b for _, b in means.values()) * 1.28
ys, h, gap = [], 0.30, 0.42
y = 0.0
for i, (w, b) in enumerate(means.values()):
    ax.barh(y, w, height=h, color=GREY)
    ax.barh(y - gap, b, height=h, color=BLUE)
    if i == 0:  # one legend, read once, on the first group only
        ax.text(w + xmax * 0.02, y, "which way it moved", fontsize=20 * PX,
                color="#8C8B85", va="center")
        ax.text(b + xmax * 0.02, y - gap, "how big the move was",
                fontsize=20 * PX, color=BLUE, va="center")
    ys.append(y - gap / 2)
    y -= 1.35
ax.set_xlim(0, xmax)
ax.set_ylim(-3.27, 0.15)  # bars flush to the axes box: top 275px, bottom 720px
ax.axis("off")
for (p, _), yc in zip(means.items(), ys, strict=True):
    ax.text(-xmax * 0.04, yc, p, fontsize=22 * PX, color=INK,
            ha="right", va="center", weight="medium")

fig.text(0.21, 0.296, "bar length = how much the past ten days help predict today",
         fontsize=20 * PX, color=MUTED)

# --- callout, right of the chart, level with the USDJPY row ---
fig.text(0.655, 0.60, "5-8x", fontsize=44 * PX, color=BLUE,
         weight="bold", ha="left")
fig.text(0.655, 0.555, "how much more the size\nof a move tells you\nthan its direction",
         fontsize=22 * PX, color=INK, ha="left", va="top", linespacing=1.45)

fig.text(0.06, 0.075, "three currency pairs · ten years of daily closes",
         fontsize=19 * PX, color=MUTED)

out = ROOT / "outputs" / "case_study" / "figures" / "past_help.png"
fig.savefig(out, dpi=100, facecolor="white")
im = Image.open(out)
bg = Image.new("RGB", im.size, (255, 255, 255))
bg.paste(im, mask=im.split()[3])
bg.save(out, optimize=True)
print("saved", out, Image.open(out).size, Image.open(out).mode)
