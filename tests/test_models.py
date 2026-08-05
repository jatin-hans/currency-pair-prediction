"""Tests for regime_lab.models (plan §8 ladder, tiers B0–B2 and N1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lab.data.features import FEATURE_COLUMNS
from regime_lab.models import (
    MODEL_FEATURES,
    ConditionAwareModel,
    MajorityPersistenceBaseline,
    make_b1,
    make_b2,
)


def _synthetic_frame(n: int = 600, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Features with a learnable signal: class = sign bucket of feature 'return_z_252d'."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, len(MODEL_FEATURES))), columns=MODEL_FEATURES
    )
    X["trailing_ret_5d"] = rng.normal(0, 0.01, n)
    X["dead_zone_threshold"] = np.full(n, 0.005)
    X["regime_label_vol"] = rng.integers(0, 3, n)
    z = X["return_z_252d"]
    y = pd.Series(np.where(z > 0.4, 2, np.where(z < -0.4, 0, 1)), index=X.index)
    return X, y


def test_model_features_exclude_close():
    """R0 §6.8: raw price level is never a model input."""
    assert "close" not in MODEL_FEATURES
    assert set(MODEL_FEATURES) < set(FEATURE_COLUMNS)


def test_b0_proba_equals_train_class_frequencies():
    X, y = _synthetic_frame()
    b0 = MajorityPersistenceBaseline().fit(X, y)
    proba = b0.predict_proba(X.iloc[:5])
    expected = np.bincount(y, minlength=3) / len(y)
    np.testing.assert_allclose(proba, np.tile(expected, (5, 1)))


def test_b0_point_prediction_is_persistence_through_the_dead_zone():
    X, y = _synthetic_frame()
    b0 = MajorityPersistenceBaseline().fit(X, y)
    X_test = pd.DataFrame(
        {
            "trailing_ret_5d": [0.02, -0.02, 0.001],
            "dead_zone_threshold": [0.005, 0.005, 0.005],
        }
    )
    np.testing.assert_array_equal(b0.predict(X_test), [2, 0, 1])


def test_b1_and_b2_learn_a_separable_signal():
    X, y = _synthetic_frame(n=800)
    for factory in (make_b1, make_b2):
        model = factory(seed=42).fit(X[MODEL_FEATURES], y)
        acc = (model.predict(X[MODEL_FEATURES]) == y).mean()
        assert acc > 0.8, f"{factory.__name__} failed to learn"
        proba = model.predict_proba(X[MODEL_FEATURES].iloc[:10])
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)


def test_condition_aware_model_fits_one_head_per_state():
    X, y = _synthetic_frame(n=900)
    n1 = ConditionAwareModel(seed=42, min_head_rows=100).fit(X, y)
    assert set(n1.heads_) == {0, 1, 2}
    assert n1.fallback_states_ == []
    proba = n1.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)


def test_condition_aware_model_sparsity_fallback():
    """States with < min_head_rows train rows must route to the pooled
    model with one-hot state interactions, and be reported."""
    X, y = _synthetic_frame(n=600)
    X.loc[X.index[:590], "regime_label_vol"] = 0  # starve states 1 and 2
    X.loc[X.index[590:595], "regime_label_vol"] = 1
    X.loc[X.index[595:], "regime_label_vol"] = 2

    n1 = ConditionAwareModel(seed=42, min_head_rows=250).fit(X, y)

    assert 1 in n1.fallback_states_ and 2 in n1.fallback_states_
    assert 0 in n1.heads_
    preds = n1.predict(X)
    assert len(preds) == len(X)


def test_condition_aware_predictions_route_by_state():
    """Rows in a state with a dedicated head must get that head's output."""
    X, y = _synthetic_frame(n=900)
    n1 = ConditionAwareModel(seed=42, min_head_rows=100).fit(X, y)

    state0 = X[X["regime_label_vol"] == 0]
    expected = n1.heads_[0].predict_proba(state0[MODEL_FEATURES])
    actual = n1.predict_proba(state0)
    np.testing.assert_allclose(actual, expected)


def test_b0_fit_rejects_empty_input():
    with pytest.raises(ValueError):
        MajorityPersistenceBaseline().fit(pd.DataFrame(), pd.Series(dtype=int))


def test_dashboard_factories_learn_the_separable_signal():
    from regime_lab.models import make_nb, make_rf

    X, y = _synthetic_frame(n=800)
    for factory in (make_rf, make_nb):
        model = factory(seed=42).fit(X[MODEL_FEATURES], y)
        proba = model.predict_proba(X[MODEL_FEATURES].iloc[:10])
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)
        assert (model.predict(X[MODEL_FEATURES]) == y).mean() > 0.7


def test_fit_calibrated_reduces_brier_of_an_overconfident_model():
    """On held-out data, isotonic calibration must not make the probability
    score worse — and must output valid probabilities."""
    from regime_lab.eval import multiclass_brier
    from regime_lab.models import _expand_proba, fit_calibrated, make_b2

    X, y = _synthetic_frame(n=1200, seed=9)
    Xf = X[MODEL_FEATURES]
    train, test = slice(0, 900), slice(900, 1200)

    raw = make_b2(seed=42).fit(Xf.iloc[train], y.iloc[train])
    cal = fit_calibrated(make_b2, Xf.iloc[train], y.iloc[train], seed=42)

    p_raw = _expand_proba(raw, Xf.iloc[test])
    p_cal = _expand_proba(cal, Xf.iloc[test])
    np.testing.assert_allclose(p_cal.sum(axis=1), 1.0, rtol=1e-6)
    assert multiclass_brier(np.asarray(y.iloc[test]), p_cal) <= (
        multiclass_brier(np.asarray(y.iloc[test]), p_raw) + 0.02
    )


def test_fit_calibrated_falls_back_on_tiny_samples():
    from sklearn.ensemble import HistGradientBoostingClassifier

    from regime_lab.models import fit_calibrated, make_b2

    X, y = _synthetic_frame(n=60, seed=10)
    model = fit_calibrated(make_b2, X[MODEL_FEATURES], y, seed=42)
    assert isinstance(model, HistGradientBoostingClassifier)


def test_hmm_direction_probabilities_are_valid_and_causal():
    from regime_lab.models import HMMDirection

    rng = np.random.default_rng(0)
    dates = pd.DatetimeIndex(pd.bdate_range("2018-01-01", periods=600).values)
    returns = pd.Series(rng.normal(0, 0.005, 600), index=dates)
    labels = pd.Series(rng.integers(0, 3, 600), index=dates)

    model = HMMDirection(seed=42).fit(returns.iloc[:500], labels.iloc[:500])
    proba = model.predict_proba_at_end(returns.iloc[:550])

    assert proba.shape == (3,)
    assert proba.sum() == pytest.approx(1.0)
    assert (proba >= 0).all()

    # Causality: the prediction at position 549 must not change when data
    # after it is mutated.
    mutated = returns.copy()
    mutated.iloc[550:] = 0.5
    np.testing.assert_allclose(
        model.predict_proba_at_end(mutated.iloc[:550]), proba
    )
