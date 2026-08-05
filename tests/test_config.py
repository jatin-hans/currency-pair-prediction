"""Tests for regime_lab.config.Settings.

Asserts the default Settings instance matches the implementation plan exactly
(spec §4.1, §4.2, §4.4, §6.6, §6.7, §6.8, §7.2).
"""

from __future__ import annotations

from datetime import date

from regime_lab.config import Settings


def test_settings_instantiates_with_defaults():
    settings = Settings()
    assert settings is not None


def test_pairs_match_spec_section_4_1():
    settings = Settings()
    assert settings.pairs == {
        "EURUSD": "EURUSD=X",
        "USDINR": "INR=X",
        "USDJPY": "JPY=X",
    }


def test_auxiliary_tickers_match_spec_section_4_1():
    settings = Settings()
    assert settings.auxiliary_tickers == {
        "DXY": "DX-Y.NYB",
        "VIX": "^VIX",
    }


def test_date_range_start_matches_spec_section_4_2():
    settings = Settings()
    assert settings.date_start == date(2015, 1, 1)


def test_seed_is_42_per_spec_section_6_8():
    settings = Settings()
    assert settings.seed == 42


def test_direction_task_params_match_plan_section_7():
    settings = Settings()
    assert settings.direction_k == 5
    assert settings.dead_zone_mult == 0.25
    assert settings.vol_window == 20


def test_conformal_and_bootstrap_params_match_plan_sections_4_and_8():
    settings = Settings()
    assert settings.target_coverage == 0.80
    assert settings.bootstrap_block_length == 10
    assert settings.bootstrap_n_resamples == 1000


def test_walk_forward_blocks_match_spec_section_4_4():
    settings = Settings()
    # 6 non-overlapping test blocks per spec §4.4.
    assert len(settings.walk_forward_blocks) == 6

    expected_test_starts = [
        date(2017, 7, 1),
        date(2018, 7, 1),
        date(2020, 1, 1),
        date(2021, 7, 1),
        date(2023, 7, 1),
        date(2025, 1, 1),
    ]
    actual_test_starts = [block.test_start for block in settings.walk_forward_blocks]
    assert actual_test_starts == expected_test_starts

    # Every train block starts at the data-range start (expanding train).
    for block in settings.walk_forward_blocks:
        assert block.train_start == settings.date_start

    # Every block is leakage-free: train_end < test_start.
    for block in settings.walk_forward_blocks:
        assert block.train_end < block.test_start

    # Every block's test range is ordered.
    for block in settings.walk_forward_blocks:
        assert block.test_start <= block.test_end


def test_regime_tercile_window_matches_spec_section_5_1():
    settings = Settings()
    assert settings.tercile_min_window == 252


def test_data_and_output_roots_are_relative_paths():
    """Spec §10.3: no hardcoded absolute paths."""
    settings = Settings()
    assert not settings.data_root.is_absolute()
    assert not settings.output_root.is_absolute()
