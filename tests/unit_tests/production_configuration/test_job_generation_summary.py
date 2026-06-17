from unittest.mock import Mock

import astropy.units as u

from simtools.production_configuration.job_generation_summary import (
    GeneratedRowSummary,
    ShowerRoundingSummary,
    format_quantity_bounds,
    log_streamed_row_summary,
    update_quantity_bounds,
)


def _job_row(**overrides):
    row = {
        "energy_min": 30 * u.GeV,
        "energy_max": 10 * u.TeV,
        "core_scatter_max": 200 * u.m,
        "view_cone_min": 0 * u.deg,
        "view_cone_max": 5 * u.deg,
    }
    row.update(overrides)
    return row


def test_update_quantity_bounds_initializes_and_updates_bounds():
    bounds = update_quantity_bounds(None, 30 * u.GeV)
    bounds = update_quantity_bounds(bounds, 0.1 * u.TeV)

    assert bounds == (30 * u.GeV, 100 * u.GeV)


def test_format_quantity_bounds_collapses_equal_values():
    assert format_quantity_bounds((5 * u.deg, 5 * u.deg)) == "5 deg"
    assert format_quantity_bounds((1 * u.deg, 3 * u.deg)) == "[1, 3] deg"


def test_generated_row_summary_tracks_count_and_ranges():
    summary = GeneratedRowSummary()

    summary.add(_job_row())
    summary.add(_job_row(energy_min=0.1 * u.TeV, view_cone_max=8 * u.deg))

    assert summary.count == 2
    assert summary.energy_min == (30 * u.GeV, 100 * u.GeV)
    assert summary.view_cone_max == (5 * u.deg, 8 * u.deg)


def test_shower_rounding_summary_logs_aggregate_warning():
    summary = ShowerRoundingSummary()
    logger = Mock()

    summary.add(2500, 1000, 3000)
    summary.add(2600, 1000, 3000)
    summary.log(logger)

    logger.warning.assert_called_once()
    assert logger.warning.call_args.args[1:] == (2, 2500, 1000, 3000)


def test_shower_rounding_summary_does_not_log_without_adjustments():
    logger = Mock()

    ShowerRoundingSummary().log(logger)

    logger.warning.assert_not_called()


def test_log_streamed_row_summary_logs_empty_generation():
    logger = Mock()

    log_streamed_row_summary(GeneratedRowSummary(), logger)

    logger.info.assert_called_once_with(
        "Generated 0 simulation rows after applying all clipping and scaling rules."
    )
