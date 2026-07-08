import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table

from simtools.io import table_handler
from simtools.production_configuration import monte_carlo_statistics_estimator

_LOAD_HISTOGRAMS = (
    "simtools.production_configuration.monte_carlo_statistics_estimator.load_trigger_histograms"
)


def _build_reference_tables():
    metadata = Table(
        rows=[
            {
                "reference_id": "ref-1",
                "production_index": 0,
                "array_name": "alpha",
                "core_scatter_max": 100.0 * u.m,
                "energy_min": 0.1 * u.TeV,
                "energy_max": 10.0 * u.TeV,
            }
        ]
    )
    bins = Table(
        rows=[
            {
                "reference_id": "ref-1",
                "angular_distance_bin_index": 0,
                "energy_bin_index": 0,
                "angular_distance_low": 0.0 * u.deg,
                "angular_distance_high": 1.0 * u.deg,
                "energy_low": 0.1 * u.TeV,
                "energy_high": 1.0 * u.TeV,
                "simulated_count": 100,
                "trigger_efficiency": 0.5,
            },
            {
                "reference_id": "ref-1",
                "angular_distance_bin_index": 0,
                "energy_bin_index": 1,
                "angular_distance_low": 0.0 * u.deg,
                "angular_distance_high": 1.0 * u.deg,
                "energy_low": 1.0 * u.TeV,
                "energy_high": 10.0 * u.TeV,
                "simulated_count": 100,
                "trigger_efficiency": 0.25,
            },
        ]
    )
    return metadata, bins


def test_resolve_effective_throw_radius_accepts_original_or_smaller_radius():
    assert monte_carlo_statistics_estimator._resolve_effective_throw_radius(100.0 * u.m).to_value(
        u.m
    ) == pytest.approx(100.0)
    assert monte_carlo_statistics_estimator._resolve_effective_throw_radius(
        100.0 * u.m, 80.0 * u.m
    ).to_value(u.m) == pytest.approx(80.0)


def test_resolve_effective_throw_radius_rejects_invalid_override():
    with pytest.raises(ValueError, match="positive"):
        monte_carlo_statistics_estimator._resolve_effective_throw_radius(100.0 * u.m, 0.0 * u.m)
    with pytest.raises(ValueError, match="cannot exceed"):
        monte_carlo_statistics_estimator._resolve_effective_throw_radius(100.0 * u.m, 120.0 * u.m)


def test_compute_effective_area_matrix_scales_with_radius():
    matrix = np.array([[0.5, 0.25]])
    original = monte_carlo_statistics_estimator._compute_effective_area_matrix(matrix, 100.0 * u.m)
    reduced = monte_carlo_statistics_estimator._compute_effective_area_matrix(matrix, 50.0 * u.m)

    np.testing.assert_allclose(reduced, original / 4.0)


def test_estimate_required_events_skips_empty_bins():
    expected_triggers_per_event = np.array([[0.0, 0.2], [0.1, 0.0]])

    required, limiting_expected_per_event, limiting_index, used_bins, skipped_bins = (
        monte_carlo_statistics_estimator._estimate_required_events(
            expected_triggers_per_event,
            np.array([True, True]),
            0.1,
        )
    )

    assert required == pytest.approx(1000.0)
    assert limiting_expected_per_event == pytest.approx(0.1)
    assert limiting_index == (1, 0)
    assert used_bins == 2
    assert skipped_bins == 2


def test_estimator_radius_override_changes_geometric_normalization_not_required_events(
    mocker, tmp_path
):
    metadata, bins = _build_reference_tables()
    mocker.patch(
        _LOAD_HISTOGRAMS,
        return_value=(metadata, bins),
    )

    common_args = {
        "input": "unused.hdf5",
        "array_names": None,
        "spectral_index": -2.0,
        "target_relative_uncertainty": 0.1,
        "br_energy_min": 0.1 * u.TeV,
        "br_energy_max": 10.0 * u.TeV,
        "optimization_energy_min": 0.1 * u.TeV,
        "optimization_energy_max": 10.0 * u.TeV,
    }

    args_original = common_args | {
        "reduced_core_radius": None,
        "output_file": str(tmp_path / "a.ecsv"),
    }
    args_reduced = common_args | {
        "reduced_core_radius": 50.0 * u.m,
        "output_file": str(tmp_path / "b.ecsv"),
    }

    original = monte_carlo_statistics_estimator.estimate_monte_carlo_statistics(args_original)
    reduced = monte_carlo_statistics_estimator.estimate_monte_carlo_statistics(args_reduced)

    assert original["estimated_total_events"][0] == pytest.approx(
        reduced["estimated_total_events"][0]
    )
    assert reduced["effective_core_scatter_radius"].quantity[0].to_value(u.m) == pytest.approx(50.0)
    assert reduced["effective_scatter_area"].quantity[0].to_value(u.m**2) == pytest.approx(
        original["effective_scatter_area"].quantity[0].to_value(u.m**2) / 4.0
    )
    assert reduced["limiting_effective_area"].quantity[0].to_value(u.m**2) == pytest.approx(
        original["limiting_effective_area"].quantity[0].to_value(u.m**2) / 4.0
    )
    assert reduced["optimization_bins_used"][0] == 2
    assert reduced["optimization_bins_skipped"][0] == 0


def test_estimator_writes_diagnostic_plots(mocker, tmp_path):
    metadata, bins = _build_reference_tables()
    mocker.patch(
        _LOAD_HISTOGRAMS,
        return_value=(metadata, bins),
    )
    mock_plot = mocker.patch(
        "simtools.production_configuration.monte_carlo_statistics_estimator."
        "plot_monte_carlo_statistics_diagnostics"
    )

    monte_carlo_statistics_estimator.estimate_monte_carlo_statistics(
        {
            "input": "unused.hdf5",
            "array_names": None,
            "spectral_index": -2.0,
            "target_relative_uncertainty": 0.1,
            "br_energy_min": 0.1 * u.TeV,
            "br_energy_max": 10.0 * u.TeV,
            "optimization_energy_min": 0.1 * u.TeV,
            "optimization_energy_max": 10.0 * u.TeV,
            "reduced_core_radius": None,
            "plot_diagnostics": True,
            "output_file": str(tmp_path / "estimate.ecsv"),
        }
    )

    mock_plot.assert_called_once()


def test_estimator_reports_limiting_bin_and_positive_required_events(mocker, tmp_path):
    metadata, bins = _build_reference_tables()
    mocker.patch(
        _LOAD_HISTOGRAMS,
        return_value=(metadata, bins),
    )

    args = {
        "input": "unused.hdf5",
        "array_names": None,
        "spectral_index": -2.0,
        "target_relative_uncertainty": 0.1,
        "br_energy_min": 0.1 * u.TeV,
        "br_energy_max": 10.0 * u.TeV,
        "optimization_energy_min": 0.1 * u.TeV,
        "optimization_energy_max": 10.0 * u.TeV,
        "reduced_core_radius": None,
        "output_file": str(tmp_path / "estimate.ecsv"),
    }

    result = monte_carlo_statistics_estimator.estimate_monte_carlo_statistics(args)

    assert result["array_name"][0] == "alpha"
    assert result["estimated_total_events"][0] > 0.0
    assert result["limiting_energy_low"].quantity[0] in (0.1 * u.TeV, 1.0 * u.TeV)


def test_estimator_supports_reference_tables_reloaded_from_hdf5(mocker, tmp_path):
    metadata, bins = _build_reference_tables()
    reference_file = tmp_path / "trigger_histograms.hdf5"
    metadata.meta["EXTNAME"] = "TRIGGER_REFERENCE_METADATA"
    bins.meta["EXTNAME"] = "TRIGGER_REFERENCE_BINS"
    table_handler.write_tables([metadata, bins], reference_file, file_type="HDF5")
    reloaded_metadata, reloaded_bins = monte_carlo_statistics_estimator.load_trigger_histograms(
        reference_file
    )
    mocker.patch(
        _LOAD_HISTOGRAMS,
        return_value=(reloaded_metadata, reloaded_bins),
    )

    args = {
        "input": str(reference_file),
        "array_names": None,
        "spectral_index": -2.0,
        "target_relative_uncertainty": 0.1,
        "br_energy_min": None,
        "br_energy_max": None,
        "optimization_energy_min": None,
        "optimization_energy_max": None,
        "reduced_core_radius": None,
        "output_file": str(tmp_path / "estimate.ecsv"),
    }

    result = monte_carlo_statistics_estimator.estimate_monte_carlo_statistics(args)

    assert result["effective_core_scatter_radius"].quantity[0].to_value(u.m) == pytest.approx(100.0)
    assert result["estimated_total_events"][0] > 0.0
