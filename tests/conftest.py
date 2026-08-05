"""Shared pytest fixtures for Regime Lab.

Kept minimal in Phase 0; more fixtures land in later phases (sample price
frames, synthetic label vectors, etc.) once the modules that need them
exist.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic RNG seeded per spec §6.8 (seed=42)."""
    return np.random.default_rng(42)


@pytest.fixture
def tmp_output_root(tmp_path: Path) -> Path:
    """Isolated output root for tests that write artifacts."""
    root = tmp_path / "outputs"
    root.mkdir()
    for sub in ("models", "predictions", "hedge", "benchmarks", "figures"):
        (root / sub).mkdir()
    return root
