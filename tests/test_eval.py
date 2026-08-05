"""Tests for regime_lab.eval (metrics + block bootstrap, plan §7)."""

from __future__ import annotations

import numpy as np
import pytest

from regime_lab.eval import (
    block_bootstrap_ci,
    multiclass_brier,
    paired_balanced_accuracy_diff_ci,
    paired_brier_diff_ci,
    per_row_brier,
    summarize_predictions,
)


def test_multiclass_brier_perfect_prediction_is_zero():
    y = np.array([0, 1, 2])
    proba = np.eye(3)[y]
    assert multiclass_brier(y, proba) == 0.0


def test_multiclass_brier_uniform_prediction_matches_hand_value():
    """Uniform 1/3 proba: each row scores (1/3)² + (1/3)² + (2/3)² = 2/3."""
    y = np.array([0, 1, 2, 0])
    proba = np.full((4, 3), 1.0 / 3.0)
    assert multiclass_brier(y, proba) == pytest.approx(2.0 / 3.0)


def test_per_row_brier_mean_equals_multiclass_brier():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 50)
    raw = rng.random((50, 3))
    proba = raw / raw.sum(axis=1, keepdims=True)
    assert per_row_brier(y, proba).mean() == pytest.approx(multiclass_brier(y, proba))


def test_summarize_predictions_returns_expected_keys():
    y = np.array([0, 1, 2, 2])
    y_pred = np.array([0, 1, 2, 0])
    proba = np.full((4, 3), 1.0 / 3.0)
    out = summarize_predictions(y, y_pred, proba)
    assert set(out) == {"n", "balanced_accuracy", "macro_f1", "brier"}
    assert out["n"] == 4


def test_block_bootstrap_ci_is_seeded_and_ordered():
    rng_values = np.random.default_rng(1).normal(0.5, 0.1, 300)

    def stat(idx: np.ndarray) -> float:
        return float(rng_values[idx].mean())

    lo1, hi1 = block_bootstrap_ci(stat, n=300, block_length=10, n_resamples=200, seed=42)
    lo2, hi2 = block_bootstrap_ci(stat, n=300, block_length=10, n_resamples=200, seed=42)
    assert (lo1, hi1) == (lo2, hi2)
    assert lo1 <= hi1
    # A percentile bootstrap CI must cover the observed sample statistic.
    assert lo1 <= rng_values.mean() <= hi1


def test_paired_brier_diff_ci_of_identical_models_is_zero():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 3, 200)
    raw = rng.random((200, 3))
    proba = raw / raw.sum(axis=1, keepdims=True)

    diff, (lo, hi) = paired_brier_diff_ci(
        y, proba, proba, block_length=10, n_resamples=100, seed=0
    )
    assert diff == 0.0
    assert (lo, hi) == (0.0, 0.0)


def test_paired_balanced_accuracy_diff_of_identical_predictions_is_zero():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 3, 300)
    pred = rng.integers(0, 3, 300)
    diff, (lo, hi) = paired_balanced_accuracy_diff_ci(
        y, pred, pred, block_length=10, n_resamples=100, seed=0
    )
    assert diff == 0.0
    assert (lo, hi) == (0.0, 0.0)


def test_paired_balanced_accuracy_diff_detects_a_better_model():
    """Model A is right 80% of the time, model B guesses: the diff must be
    positive with a CI excluding zero."""
    rng = np.random.default_rng(5)
    y = rng.integers(0, 3, 600)
    pred_a = np.where(rng.random(600) < 0.8, y, rng.integers(0, 3, 600))
    pred_b = rng.integers(0, 3, 600)

    diff, (lo, hi) = paired_balanced_accuracy_diff_ci(
        y, pred_a, pred_b, block_length=10, n_resamples=300, seed=0
    )
    assert diff > 0.3
    assert lo > 0


def test_paired_brier_diff_ci_detects_a_strictly_better_model():
    """Model A predicts the truth sharply; model B is uniform. The diff
    Brier(A) − Brier(B) must be negative with a CI excluding zero."""
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, 400)
    proba_a = np.eye(3)[y] * 0.94 + 0.02
    proba_b = np.full((400, 3), 1.0 / 3.0)

    diff, (lo, hi) = paired_brier_diff_ci(
        y, proba_a, proba_b, block_length=10, n_resamples=500, seed=0
    )
    assert diff < 0
    assert hi < 0
