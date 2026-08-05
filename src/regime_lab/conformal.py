"""State-conditional (Mondrian) split conformal prediction sets.

Score: s = 1 − p̂_y (one minus the probability the classifier gave the true
class). Calibration computes a per-state score quantile q̂; the prediction
set at a test point keeps every class c with p̂_c ≥ 1 − q̂(state). The
selective predictor answers only when the set is a single class.
Caveat: FX days are not exchangeable, so the split-conformal coverage
guarantee does not transfer — every reported coverage is empirical.
"""

from __future__ import annotations

import math

import numpy as np

N_CLASSES = 3


def conformal_quantile(scores: np.ndarray, target_coverage: float) -> float:
    """Finite-sample-corrected quantile: the ⌈(n+1)·coverage⌉-th smallest
    score (clamped to the maximum when n is too small for the correction)."""
    n = len(scores)
    if n == 0:
        raise ValueError("cannot calibrate on zero scores")
    k = min(n, math.ceil((n + 1) * target_coverage))
    return float(np.sort(np.asarray(scores))[k - 1])


class MondrianConformal:
    """Per-volatility-state split conformal over a fitted probabilistic
    classifier's outputs. States with fewer than ``min_state_calib``
    calibration rows fall back to the marginal quantile (recorded in
    ``fallback_states_``)."""

    def __init__(self, *, target_coverage: float = 0.80, min_state_calib: int = 30) -> None:
        self.target_coverage = target_coverage
        self.min_state_calib = min_state_calib

    def calibrate(
        self, proba: np.ndarray, y: np.ndarray, states: np.ndarray
    ) -> MondrianConformal:
        y = np.asarray(y, dtype=int)
        states = np.asarray(states, dtype=int)
        scores = 1.0 - proba[np.arange(len(y)), y]

        self.marginal_qhat_ = conformal_quantile(scores, self.target_coverage)
        self.state_qhats_: dict[int, float] = {}
        self.fallback_states_: list[int] = []
        for state in np.unique(states):
            mask = states == state
            if int(mask.sum()) >= self.min_state_calib:
                self.state_qhats_[int(state)] = conformal_quantile(
                    scores[mask], self.target_coverage
                )
            else:
                self.fallback_states_.append(int(state))
        return self

    def qhat_for(self, state: int) -> float:
        return self.state_qhats_.get(int(state), self.marginal_qhat_)

    def predict_sets(self, proba: np.ndarray, states: np.ndarray) -> np.ndarray:
        """Boolean (n, 3) membership matrix of the prediction sets."""
        qhats = np.array([self.qhat_for(s) for s in np.asarray(states, dtype=int)])
        return proba >= (1.0 - qhats)[:, None]
