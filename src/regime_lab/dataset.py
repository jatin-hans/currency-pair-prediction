"""Dataset assembly for the direction task.

Joins the processed feature parquet with the direction target and the two
decision-time columns B0 needs (trailing 5d return, dead-zone threshold).
Every added column is trailing-only; the target is the only thing that
looks forward, and that is its job.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regime_lab.config import Settings
from regime_lab.data.labels import label_direction

TARGET_COLUMN = "direction_label"


def assemble_dataset(processed: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Features + volatility state + B0 columns + direction target."""
    df = processed.copy()
    log_close = np.log(df["close"].astype(float))

    df["trailing_ret_5d"] = log_close.diff(settings.direction_k)
    trailing_vol = log_close.diff().rolling(settings.vol_window).std()
    df["dead_zone_threshold"] = (
        settings.dead_zone_mult * trailing_vol * np.sqrt(settings.direction_k)
    )

    target = label_direction(
        df["close"],
        k=settings.direction_k,
        dead_zone_mult=settings.dead_zone_mult,
        vol_window=settings.vol_window,
    )
    df = df.join(target, how="inner").dropna()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype("int64")
    return df


def load_pair_dataset(
    pair: str, settings: Settings, root: Path = Path(".")
) -> pd.DataFrame:
    processed = pd.read_parquet(
        Path(root) / settings.data_root / "processed" / f"{pair}.parquet"
    )
    return assemble_dataset(processed, settings)
