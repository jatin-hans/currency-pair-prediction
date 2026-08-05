"""Shared driver for the result scripts (03–05): fit/predict per tier,
long-format prediction collection, metric and paired-difference tables.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regime_lab.config import Settings
from regime_lab.data.splits import split_by_block
from regime_lab.dataset import TARGET_COLUMN, load_pair_dataset
from regime_lab.eval import (
    paired_balanced_accuracy_diff_ci,
    paired_brier_diff_ci,
    summarize_predictions,
)
from regime_lab.models import (
    MODEL_FEATURES,
    ConditionAwareModel,
    MajorityPersistenceBaseline,
    _expand_proba,
    make_b1,
    make_b2,
)


def fit_predict(
    name: str, train: pd.DataFrame, test: pd.DataFrame, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Returns (point predictions, 3-class probabilities, extra info)."""
    y_train = train[TARGET_COLUMN]
    if name == "B0":
        model = MajorityPersistenceBaseline().fit(train, y_train)
        return model.predict(test), model.predict_proba(test), {}
    if name == "N1":
        model = ConditionAwareModel(seed=seed).fit(train, y_train)
        return (
            model.predict(test),
            model.predict_proba(test),
            {"fallback_states": model.fallback_states_},
        )
    factory = make_b1 if name == "B1" else make_b2
    model = factory(seed).fit(train[MODEL_FEATURES], y_train)
    proba = _expand_proba(model, test[MODEL_FEATURES])
    return proba.argmax(axis=1), proba, {}


def run_models(
    settings: Settings, model_names: list[str], root: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-day long-format predictions + per-fit info rows."""
    rows: list[dict[str, object]] = []
    info_rows: list[dict[str, object]] = []
    for pair in settings.pairs:
        df = load_pair_dataset(pair, settings, root)
        for block in settings.walk_forward_blocks:
            train, test = split_by_block(df, block, purge=settings.direction_k)
            for name in model_names:
                pred, proba, info = fit_predict(name, train, test, settings.seed)
                if info:
                    info_rows.append(
                        {"pair": pair, "block": block.block_id, "model": name} | info
                    )
                for i, (date, row) in enumerate(test.iterrows()):
                    rows.append(
                        {
                            "pair": pair,
                            "block": block.block_id,
                            "model": name,
                            "date": date,
                            "y_true": int(row[TARGET_COLUMN]),
                            "y_pred": int(pred[i]),
                            "p_down": proba[i, 0],
                            "p_flat": proba[i, 1],
                            "p_up": proba[i, 2],
                            "state": int(row["regime_label_vol"]),
                        }
                    )
        print(f"  {pair}: done ({len(model_names)} models × 6 blocks)")
    return pd.DataFrame(rows), pd.DataFrame(info_rows)


def scope_of(block_id: int) -> str:
    return "block6_confirmation" if block_id == 6 else "blocks1-5_exploratory"


def metrics_table(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    preds = preds.assign(scope=preds["block"].map(scope_of))
    for (pair, model, scope), g in preds.groupby(["pair", "model", "scope"]):
        proba = g[["p_down", "p_flat", "p_up"]].to_numpy()
        rows.append(
            {"pair": pair, "model": model, "scope": scope}
            | summarize_predictions(g["y_true"].to_numpy(), g["y_pred"].to_numpy(), proba)
        )
    return pd.DataFrame(rows).sort_values(["pair", "scope", "model"])


def _proba_matrix(g: pd.DataFrame, model: str) -> np.ndarray:
    return np.column_stack(
        [g[(c, model)].to_numpy() for c in ["p_down", "p_flat", "p_up"]]
    )


def paired_diff_table(
    preds: pd.DataFrame, comparisons: list[tuple[str, str]], settings: Settings
) -> pd.DataFrame:
    """Brier(A) − Brier(B) per pair × scope with block-bootstrap CIs."""
    rows = []
    preds = preds.assign(scope=preds["block"].map(scope_of))
    for (pair, scope), g in preds.groupby(["pair", "scope"]):
        wide = g.pivot_table(
            index="date",
            columns="model",
            values=["p_down", "p_flat", "p_up", "y_true", "y_pred"],
        ).sort_index()
        for a, b in comparisons:
            y = wide[("y_true", a)].to_numpy().astype(int)
            diff, (lo, hi) = paired_brier_diff_ci(
                y,
                _proba_matrix(wide, a),
                _proba_matrix(wide, b),
                block_length=settings.bootstrap_block_length,
                n_resamples=settings.bootstrap_n_resamples,
                seed=settings.seed,
            )
            ba_diff, (ba_lo, ba_hi) = paired_balanced_accuracy_diff_ci(
                y,
                wide[("y_pred", a)].to_numpy(),
                wide[("y_pred", b)].to_numpy(),
                block_length=settings.bootstrap_block_length,
                n_resamples=settings.bootstrap_n_resamples,
                seed=settings.seed,
            )
            rows.append(
                {
                    "pair": pair,
                    "scope": scope,
                    "comparison": f"{a}-{b}",
                    "balacc_diff": ba_diff,
                    "balacc_ci_lo": ba_lo,
                    "balacc_ci_hi": ba_hi,
                    "balacc_excludes_zero": bool(ba_hi < 0 or ba_lo > 0),
                    "brier_diff": diff,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "excludes_zero": bool(hi < 0 or lo > 0),
                }
            )
    return pd.DataFrame(rows)
