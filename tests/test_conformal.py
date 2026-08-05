"""Tests for regime_lab.conformal (Mondrian split conformal, plan §8 N3)."""

from __future__ import annotations

import numpy as np
import pytest

from regime_lab.conformal import MondrianConformal, conformal_quantile


def _noisy_proba(y: np.ndarray, sharpness: float, rng) -> np.ndarray:
    """Probabilities that put ``sharpness`` mass on the true class."""
    proba = rng.dirichlet(np.ones(3), size=len(y))
    onehot = np.eye(3)[y]
    proba = (1 - sharpness) * proba + sharpness * onehot
    return proba / proba.sum(axis=1, keepdims=True)


def test_conformal_quantile_matches_hand_computation():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    # n=9, coverage 0.8 → level ⌈10·0.8⌉/9 = 8/9 → the 8th order statistic.
    assert conformal_quantile(scores, 0.80) == pytest.approx(0.8)


def test_conformal_quantile_rejects_empty():
    with pytest.raises(ValueError):
        conformal_quantile(np.array([]), 0.8)


def test_marginal_coverage_on_exchangeable_data():
    """On iid data the empirical test coverage must land near the target."""
    rng = np.random.default_rng(0)
    y_cal = rng.integers(0, 3, 2000)
    y_test = rng.integers(0, 3, 2000)
    states_cal = rng.integers(0, 3, 2000)
    states_test = rng.integers(0, 3, 2000)
    proba_cal = _noisy_proba(y_cal, 0.4, rng)
    proba_test = _noisy_proba(y_test, 0.4, rng)

    conf = MondrianConformal(target_coverage=0.80).calibrate(proba_cal, y_cal, states_cal)
    sets = conf.predict_sets(proba_test, states_test)
    coverage = sets[np.arange(len(y_test)), y_test].mean()
    assert 0.77 <= coverage <= 0.86


def test_sharp_classifier_yields_singletons_at_roughly_the_target_rate():
    """A near-perfect classifier at 80% target answers with a singleton on
    ~80% of days; the rest are (correctly) empty sets — abstentions. It
    must never need a multi-class set."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 500)
    states = rng.integers(0, 3, 500)
    proba = _noisy_proba(y, 0.98, rng)

    conf = MondrianConformal(target_coverage=0.80).calibrate(proba, y, states)
    sets = conf.predict_sets(proba, states)
    assert (sets.sum(axis=1) <= 1).all()
    assert (sets.sum(axis=1) == 1).mean() > 0.70


def test_mondrian_thresholds_differ_when_states_differ():
    """State 0 gets sharp predictions, state 1 noisy ones → state 1 must
    carry a larger score quantile (bigger sets)."""
    rng = np.random.default_rng(2)
    y = rng.integers(0, 3, 2000)
    states = np.repeat([0, 1], 1000)
    proba = np.vstack(
        [_noisy_proba(y[:1000], 0.9, rng), _noisy_proba(y[1000:], 0.1, rng)]
    )

    conf = MondrianConformal(target_coverage=0.80).calibrate(proba, y, states)
    assert conf.qhat_for(1) > conf.qhat_for(0)


def test_sparse_state_falls_back_to_marginal():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, 300)
    states = np.zeros(300, dtype=int)
    states[:5] = 2  # state 2 has only 5 calibration rows
    proba = _noisy_proba(y, 0.5, rng)

    conf = MondrianConformal(target_coverage=0.80, min_state_calib=30).calibrate(
        proba, y, states
    )
    assert 2 in conf.fallback_states_
    assert conf.qhat_for(2) == conf.marginal_qhat_
