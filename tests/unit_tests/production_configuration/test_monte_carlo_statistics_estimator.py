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
                "primary_particle": "gamma",
                "zenith": 20.0 * u.deg,
                "azimuth": 180.0 * u.deg,
                "nsb_level": 1.0,
                "core_scatter_max": 100.0 * u.m,
                "viewcone_max": 10.0 * u.deg,
                "energy_min": 0.1 * u.TeV,
                "energy_max": 10.0 * u.TeV,
            }
        ]
    )
    rows = []
    for energy_index, (energy_low, energy_high) in enumerate([(0.1, 1.0), (1.0, 10.0)]):
        for core_index, (core_low, core_high, simulated, triggered) in enumerate(
            [(0.0, 50.0, 50, 40), (50.0, 100.0, 50, 10)]
        ):
            rows.append(
                {
                    "reference_id": "ref-1",
                    "angular_distance_bin_index": 0,
                    "energy_bin_index": energy_index,
                    "core_distance_bin_index": core_index,
                    "angular_distance_low": 0.0 * u.deg,
                    "angular_distance_high": 1.0 * u.deg,
                    "energy_low": energy_low * u.TeV,
                    "energy_high": energy_high * u.TeV,
                    "core_distance_low": core_low * u.m,
                    "core_distance_high": core_high * u.m,
                    "simulated_count": simulated,
                    "triggered_count": triggered if energy_index == 0 else triggered / 2,
                    "trigger_efficiency": 0.0,
                }
            )
    bins = Table(rows=rows)
    return metadata, bins


def _build_legacy_reference_tables():
    metadata = Table(
        rows=[
            {
                "reference_id": "ref-1",
                "production_index": 0,
                "array_name": "alpha",
                "primary_particle": "gamma",
                "zenith": 20.0 * u.deg,
                "azimuth": 180.0 * u.deg,
                "nsb_level": 1.0,
                "core_scatter_max": 100.0 * u.m,
                "viewcone_max": 10.0 * u.deg,
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
                "triggered_count": 50,
                "trigger_efficiency": 0.5,
            }
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


def test_compute_core_distance_weights_uses_area_fraction():
    edges = np.array([0.0, 50.0, 100.0])

    full = monte_carlo_statistics_estimator._compute_core_distance_weights(edges, 100.0 * u.m)
    reduced = monte_carlo_statistics_estimator._compute_core_distance_weights(edges, 75.0 * u.m)

    np.testing.assert_allclose(full, [1.0, 1.0])
    np.testing.assert_allclose(reduced, [1.0, (75.0**2 - 50.0**2) / (100.0**2 - 50.0**2)])


def test_collapse_core_distance_counts_applies_radial_weights():
    simulated = np.array([[[10.0, 30.0]]])
    triggered = np.array([[[5.0, 15.0]]])

    simulated_reduced, triggered_reduced = (
        monte_carlo_statistics_estimator._collapse_core_distance_counts(
            simulated,
            triggered,
            np.array([0.0, 50.0, 100.0]),
            50.0 * u.m,
        )
    )

    np.testing.assert_allclose(simulated_reduced, [[10.0]])
    np.testing.assert_allclose(triggered_reduced, [[5.0]])


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


def test_estimator_radius_override_changes_required_events(mocker, tmp_path):
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

    assert original["estimated_total_events"][0] != pytest.approx(
        reduced["estimated_total_events"][0]
    )
    assert reduced["effective_core_scatter_radius"].quantity[0].to_value(u.m) == pytest.approx(50.0)
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
    assert result["primary_particle"][0] == "gamma"
    assert result["zenith"].quantity[0].to_value(u.deg) == pytest.approx(20.0)
    assert result["azimuth"].quantity[0].to_value(u.deg) == pytest.approx(180.0)
    assert result["nsb_level"][0] == pytest.approx(1.0)
    assert result["estimated_total_events"][0] > 0.0
    assert result["limiting_energy_low"].quantity[0] in (0.1 * u.TeV, 1.0 * u.TeV)


def test_estimator_rejects_reduced_radius_for_legacy_2d_histograms(mocker, tmp_path):
    metadata, bins = _build_legacy_reference_tables()
    mocker.patch(
        _LOAD_HISTOGRAMS,
        return_value=(metadata, bins),
    )

    with pytest.raises(ValueError, match="Core-distance-binned trigger histograms are required"):
        monte_carlo_statistics_estimator.estimate_monte_carlo_statistics(
            {
                "input": "unused.hdf5",
                "array_names": None,
                "spectral_index": -2.0,
                "target_relative_uncertainty": 0.1,
                "optimization_energy_min": 0.1 * u.TeV,
                "optimization_energy_max": 1.0 * u.TeV,
                "reduced_core_radius": 50.0 * u.m,
                "output_file": str(tmp_path / "estimate.ecsv"),
            }
        )


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


def test_estimator_selects_reloaded_hdf5_rows_by_array_layout_name(mocker, tmp_path):
    metadata, bins = _build_reference_tables()
    metadata["array_name"] = ["CTAO-North-Alpha"]
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

    result = monte_carlo_statistics_estimator.estimate_monte_carlo_statistics(
        {
            "input": str(reference_file),
            "array_layout_name": ["CTAO-North-Alpha"],
            "spectral_index": -2.0,
            "target_relative_uncertainty": 0.1,
            "br_energy_min": None,
            "br_energy_max": None,
            "optimization_energy_min": None,
            "optimization_energy_max": None,
            "reduced_core_radius": None,
            "output_file": str(tmp_path / "estimate.ecsv"),
        }
    )

    assert result["array_name"][0] == "CTAO-North-Alpha"
