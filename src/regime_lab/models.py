"""The methods ladder: B0, B1, B2 and the condition-aware N1.

Public names for write-ups: majority-class + persistence baseline (B0),
logistic regression (B1), gradient boosting (B2), condition-aware model
(N1). All hyperparameters are fixed and modest — no hyperparameter search.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from regime_lab.data.features import FEATURE_COLUMNS

# Raw price level is non-stationary and never a model input.
MODEL_FEATURES: list[str] = [c for c in FEATURE_COLUMNS if c != "close"]

STATE_COLUMN = "regime_label_vol"
N_CLASSES = 3


def _expand_proba(model, X: pd.DataFrame) -> np.ndarray:
    """predict_proba padded to all 3 classes (a head trained on a state
    subset may never have seen one of the classes)."""
    proba = model.predict_proba(X)
    out = np.zeros((len(X), N_CLASSES))
    out[:, np.asarray(model.classes_, dtype=int)] = proba
    return out


class MajorityPersistenceBaseline:
    """B0 floor. Point prediction: persistence — the direction of the
    trailing 5-day return, pushed through the same dead zone as the label.
    Probabilities: train-window class frequencies (otherwise Brier would be
    undefined for this tier)."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MajorityPersistenceBaseline:
        if len(y) == 0:
            raise ValueError("B0 requires non-empty training data")
        self.class_freqs_ = np.bincount(np.asarray(y, dtype=int), minlength=N_CLASSES) / len(y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.tile(self.class_freqs_, (len(X), 1))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        r = X["trailing_ret_5d"].to_numpy()
        t = X["dead_zone_threshold"].to_numpy()
        return np.where(r > t, 2, np.where(r < -t, 0, 1))


def make_b1(seed: int) -> Pipeline:
    """B1: multinomial logistic regression, standardized features, no tuning."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
    )


def make_b2(seed: int) -> HistGradientBoostingClassifier:
    """B2: gradient boosting with a fixed modest budget."""
    return HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.1,
        min_samples_leaf=50,
        l2_regularization=1.0,
        random_state=seed,
    )


def fit_calibrated(factory, X: pd.DataFrame, y: pd.Series, seed: int):
    """Fit ``factory``'s model on the earlier 75% of rows and isotonic-
    calibrate its probabilities on the later 25% (time-ordered split — no
    shuffling). Falls back to the uncalibrated model when the calibration
    slice is too small or misses a class (tiny monthly-bar samples)."""
    from sklearn.calibration import CalibratedClassifierCV

    cut = int(len(X) * 0.75)
    base = factory(seed).fit(X.iloc[:cut], y.iloc[:cut])
    X_cal, y_cal = X.iloc[cut:], y.iloc[cut:]
    if len(y_cal) < 30 or y_cal.nunique() < 2:
        base.was_calibrated_ = False  # reported in the dashboard payload
        return base
    try:
        from sklearn.frozen import FrozenEstimator

        cal = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    except ImportError:  # sklearn < 1.6
        cal = CalibratedClassifierCV(base, cv="prefit", method="isotonic")
    cal = cal.fit(X_cal, y_cal)
    cal.was_calibrated_ = True
    return cal


def make_rf(seed: int) -> RandomForestClassifier:
    """Random forest, fixed modest budget (dashboard tier)."""
    return RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=20,
        random_state=seed,
        n_jobs=-1,
    )


def make_nb(seed: int) -> Pipeline:  # noqa: ARG001 - uniform factory signature
    """Gaussian naive Bayes on standardized features (dashboard tier)."""
    return make_pipeline(StandardScaler(), GaussianNB())


class HMMDirection:
    """Hidden Markov model tier (dashboard): a Gaussian HMM on daily log
    returns learns latent market states; direction probabilities are the
    posterior-weighted class frequencies observed per state in training.

    Causality: predictions at date t filter the return sequence only up to
    t (the forward pass on a prefix), never the smoothed full-sequence
    posterior, which would use future observations.
    """

    def __init__(self, *, n_states: int = 3, seed: int = 42) -> None:
        self.n_states = n_states
        self.seed = seed

    def fit(self, returns: pd.Series, labels: pd.Series) -> HMMDirection:
        from hmmlearn.hmm import GaussianHMM

        x = returns.to_numpy().reshape(-1, 1)
        self.hmm_ = GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=50,
            random_state=self.seed,
        ).fit(x)

        gamma = self.hmm_.predict_proba(x)
        aligned = labels.reindex(returns.index)
        has_label = aligned.notna().to_numpy()
        onehot = np.eye(N_CLASSES)[aligned[has_label].astype(int)]
        counts = gamma[has_label].T @ onehot + 1.0  # Laplace smoothing
        self.state_class_ = counts / counts.sum(axis=1, keepdims=True)
        return self

    def predict_proba_at_end(self, returns: pd.Series) -> np.ndarray:
        """Direction probabilities for the final date of ``returns``."""
        gamma = self.hmm_.predict_proba(returns.to_numpy().reshape(-1, 1))
        return gamma[-1] @ self.state_class_


class ConditionAwareModel:
    """N1: one B2 head per volatility state.

    Sparsity rule: a state with fewer than ``min_head_rows`` training rows
    gets no head; its rows route to a pooled B2 trained on all rows with
    one-hot state columns appended (the tree model can then form state
    interactions on its own). ``fallback_states_`` records which states
    fired the rule, per block.
    """

    def __init__(self, *, seed: int = 42, min_head_rows: int = 250) -> None:
        self.seed = seed
        self.min_head_rows = min_head_rows

    @staticmethod
    def _with_state_onehot(X: pd.DataFrame) -> pd.DataFrame:
        out = X[MODEL_FEATURES].copy()
        for state in range(N_CLASSES):
            out[f"state_{state}"] = (X[STATE_COLUMN] == state).astype(float)
        return out

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ConditionAwareModel:
        states = X[STATE_COLUMN]
        self.heads_: dict[int, HistGradientBoostingClassifier] = {}
        self.fallback_states_: list[int] = []

        for state in sorted(states.unique()):
            mask = states == state
            if int(mask.sum()) >= self.min_head_rows:
                self.heads_[int(state)] = make_b2(self.seed).fit(
                    X.loc[mask, MODEL_FEATURES], y.loc[mask]
                )
            else:
                self.fallback_states_.append(int(state))

        # Always fit the pooled fallback: it serves sparse states now and
        # any state value unseen in training later.
        self.pooled_ = make_b2(self.seed).fit(self._with_state_onehot(X), y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        out = np.empty((len(X), N_CLASSES))
        routed = np.zeros(len(X), dtype=bool)
        states = X[STATE_COLUMN].to_numpy()

        for state, head in self.heads_.items():
            mask = states == state
            if mask.any():
                out[mask] = _expand_proba(head, X.loc[mask, MODEL_FEATURES])
                routed |= mask

        if not routed.all():
            rest = ~routed
            out[rest] = _expand_proba(self.pooled_, self._with_state_onehot(X.loc[rest]))
        return out

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)
