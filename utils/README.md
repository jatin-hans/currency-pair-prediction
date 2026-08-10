# utils

One-off scripts for assets that live outside the case study — social posts,
standalone exports. Nothing here feeds the notebooks, the pipeline in
`scripts/`, or the case-study page; deleting this folder would not change a
single published number.

| Script | Objective |
|---|---|
| `linkedin_size_vs_direction_card.py` | 1080x1080 LinkedIn card: size memory vs direction memory per pair, "5-8x" callout (`outputs/case_study/figures/past_help.png`) |
| `linkedin_direction_charts.py` | Four LinkedIn SVGs of the EURUSD direction results: dots by prediction, confusion grids, cumulative edge, hit/miss raster |
| `linkedin_calibration_curve.py` | Standalone hand-built calibration-curve SVG (and its CSV) from the held-out EURUSD predictions |

Run any of them from the repo root with `uv run python utils/<script>.py`
(the card also needs `--with pillow`). Each recomputes its numbers from the
stored data — nothing is read off existing charts or hardcoded.
