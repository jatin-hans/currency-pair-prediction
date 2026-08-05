"""Metrics and uncertainty for the direction task.

All CIs are moving-block bootstrap over test days (block length per
``Settings.bootstrap_block_length``) — daily FX outcomes are serially
dependent, so an iid bootstrap would understate uncertainty. Comparisons
between models use the paired-difference bootstrap: resample days once,
evaluate both models on the same resample.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

N_CLASSES = 3


def per_row_brier(y: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """Per-row multiclass Brier contributions: Σ_c (p_c − 1[y=c])²."""
    y = np.asarray(y, dtype=int)
    onehot = np.eye(N_CLASSES)[y]
    return ((np.asarray(proba) - onehot) ** 2).sum(axis=1)


def multiclass_brier(y: np.ndarray, proba: np.ndarray) -> float:
    return float(per_row_brier(y, proba).mean())


def summarize_predictions(
    y: np.ndarray, y_pred: np.ndarray, proba: np.ndarray
) -> dict[str, float]:
    return {
        "n": int(len(y)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "brier": multiclass_brier(y, proba),
    }


def block_bootstrap_ci(
    stat_fn: Callable[[np.ndarray], float],
    *,
    n: int,
    block_length: int,
    n_resamples: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI for ``stat_fn`` under a moving-block bootstrap.

    ``stat_fn`` receives an index array into the original n rows and
    returns a scalar. Blocks of consecutive indices preserve short-range
    serial dependence inside each resample.
    """
    if n < 1:
        raise ValueError("n must be ≥ 1")
    block_length = min(block_length, n)
    n_blocks = int(np.ceil(n / block_length))
    rng = np.random.default_rng(seed)

    stats = np.empty(n_resamples)
    for i in range(n_resamples):
        starts = rng.integers(0, n - block_length + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block_length)[None, :]).ravel()[:n]
        stats[i] = stat_fn(idx)

    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_balanced_accuracy_diff_ci(
    y: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    *,
    block_length: int,
    n_resamples: int,
    seed: int,
) -> tuple[float, tuple[float, float]]:
    """BalancedAcc(A) − BalancedAcc(B) with a block-bootstrap CI.

    Positive = model A more accurate. The headline-friendly companion to
    the Brier difference: same paired-resample design, plain-accuracy
    units. A resample missing a class averages over the classes present.
    """
    y = np.asarray(y, dtype=int)
    a = np.asarray(pred_a, dtype=int)
    b = np.asarray(pred_b, dtype=int)

    def stat(idx: np.ndarray) -> float:
        return float(
            balanced_accuracy_score(y[idx], a[idx])
            - balanced_accuracy_score(y[idx], b[idx])
        )

    diff = float(balanced_accuracy_score(y, a) - balanced_accuracy_score(y, b))
    ci = block_bootstrap_ci(
        stat, n=len(y), block_length=block_length, n_resamples=n_resamples, seed=seed
    )
    return diff, ci


def paired_brier_diff_ci(
    y: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    *,
    block_length: int,
    n_resamples: int,
    seed: int,
) -> tuple[float, tuple[float, float]]:
    """Brier(A) − Brier(B) with a block-bootstrap CI on the difference.

    Negative = model A has lower (better) Brier. This paired difference
    is the only form used for headline comparisons.
    """
    d = per_row_brier(y, proba_a) - per_row_brier(y, proba_b)
    ci = block_bootstrap_ci(
        lambda idx: float(d[idx].mean()),
        n=len(d),
        block_length=block_length,
        n_resamples=n_resamples,
        seed=seed,
    )
    return float(d.mean()), ci
