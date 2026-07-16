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
                "spectral_index": -2.0,
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
                "spectral_index": -2.0,
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


def _build_reference_tables_with_multiple_angular_bins():
    metadata, _ = _build_reference_tables()
    bins = Table(
        rows=[
            {
                "reference_id": "ref-1",
                "angular_distance_bin_index": angular_index,
                "energy_bin_index": energy_index,
                "angular_distance_low": angular_low * u.deg,
                "angular_distance_high": angular_high * u.deg,
                "energy_low": energy_low * u.TeV,
                "energy_high": energy_high * u.TeV,
                "simulated_count": simulated,
                "triggered_count": triggered,
                "trigger_efficiency": 0.0,
            }
            for angular_index, (angular_low, angular_high, simulated, triggered) in enumerate(
                [(0.0, 5.0, 70, 35), (5.0, 10.0, 30, 3)]
            )
            for energy_index, (energy_low, energy_high) in enumerate([(0.1, 1.0), (1.0, 10.0)])
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


def test_resolve_effective_view_cone_radius_accepts_original_or_smaller_radius():
    assert monte_carlo_statistics_estimator._resolve_effective_view_cone_radius(
        10.0 * u.deg
    ).to_value(u.deg) == pytest.approx(10.0)
    assert monte_carlo_statistics_estimator._resolve_effective_view_cone_radius(
        10.0 * u.deg, 2.0 * u.deg
    ).to_value(u.deg) == pytest.approx(2.0)


def test_resolve_effective_throw_radius_rejects_invalid_override():
    with pytest.raises(ValueError, match="positive"):
        monte_carlo_statistics_estimator._resolve_effective_throw_radius(100.0 * u.m, 0.0 * u.m)
    with pytest.raises(ValueError, match="cannot exceed"):
        monte_carlo_statistics_estimator._resolve_effective_throw_radius(100.0 * u.m, 120.0 * u.m)


def test_resolve_effective_view_cone_radius_rejects_invalid_override():
    with pytest.raises(ValueError, match="positive"):
        monte_carlo_statistics_estimator._resolve_effective_view_cone_radius(
            10.0 * u.deg, 0.0 * u.deg
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        monte_carlo_statistics_estimator._resolve_effective_view_cone_radius(
            10.0 * u.deg, 12.0 * u.deg
        )


def test_compute_core_distance_weights_uses_area_fraction():
    edges = np.array([0.0, 50.0, 100.0])

    full = monte_carlo_statistics_estimator._compute_core_distance_weights(edges, 100.0 * u.m)
    reduced = monte_carlo_statistics_estimator._compute_core_distance_weights(edges, 75.0 * u.m)

    np.testing.assert_allclose(full, [1.0, 1.0])
    np.testing.assert_allclose(reduced, [1.0, (75.0**2 - 50.0**2) / (100.0**2 - 50.0**2)])


def test_compute_view_cone_weights_uses_solid_angle_fraction():
    edges = np.array([0.0, 5.0, 10.0])

    full = monte_carlo_statistics_estimator._compute_view_cone_weights(edges, 10.0 * u.deg)
    reduced = monte_carlo_statistics_estimator._compute_view_cone_weights(edges, 7.5 * u.deg)

    np.testing.assert_allclose(full, [1.0, 1.0])
    np.testing.assert_allclose(
        reduced,
        [
            1.0,
            (np.cos(np.deg2rad(5.0)) - np.cos(np.deg2rad(7.5)))
            / (np.cos(np.deg2rad(5.0)) - np.cos(np.deg2rad(10.0))),
        ],
    )


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


def test_restrict_view_cone_counts_applies_angular_weights():
    simulated = np.array([[10.0], [30.0]])
    triggered = np.array([[5.0], [15.0]])

    simulated_reduced, triggered_reduced = (
        monte_carlo_statistics_estimator._restrict_view_cone_counts(
            simulated,
            triggered,
            np.array([0.0, 5.0, 10.0]),
            5.0 * u.deg,
        )
    )

    np.testing.assert_allclose(simulated_reduced, [[10.0], [0.0]])
    np.testing.assert_allclose(triggered_reduced, [[5.0], [0.0]])


def test_estimate_required_events_skips_empty_bins():
    expected_triggers_per_event = np.array([[0.0, 0.2], [0.1, 0.0]])

    required = monte_carlo_statistics_estimator._estimate_required_events(
        expected_triggers_per_event,
        np.array([True, True]),
        0.1,
    )

    assert required == pytest.approx(1000.0)


def test_estimate_required_events_supports_target_triggered_events():
    expected_triggers_per_event = np.array([[0.0, 0.2], [0.1, 0.0]])

    required = monte_carlo_statistics_estimator._estimate_required_events(
        expected_triggers_per_event,
        np.array([True, True]),
        target_triggered_events=25,
        overall_trigger_probability=0.15,
    )

    assert required == pytest.approx(25.0 / 0.15)


def test_compute_overall_trigger_probability_uses_weighted_expected_triggers():
    expected_triggers_per_event = np.array([[0.9, 0.0], [0.0, 0.0]])

    probability = monte_carlo_statistics_estimator._compute_overall_trigger_probability(
        expected_triggers_per_event,
        np.array([True, True]),
    )

    assert probability == pytest.approx(0.9)


def test_ceil_required_total_events_rounds_up_to_integer():
    assert monte_carlo_statistics_estimator._ceil_required_total_events(1000.0) == 1000
    assert monte_carlo_statistics_estimator._ceil_required_total_events(1000.1) == 1001
    assert np.isinf(monte_carlo_statistics_estimator._ceil_required_total_events(np.inf))


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
        "target_triggered_events": None,
        "br_energy_min": 0.1 * u.TeV,
        "br_energy_max": 10.0 * u.TeV,
        "optimization_energy_min": 0.1 * u.TeV,
        "optimization_energy_max": 10.0 * u.TeV,
        "reduced_view_cone_radius": None,
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
    assert reduced.meta["reduced_core_radius"].to_value(u.m) == pytest.approx(50.0)
    assert "effective_core_scatter_radius" not in reduced.colnames


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
            "target_triggered_events": None,
            "br_energy_min": 0.1 * u.TeV,
            "br_energy_max": 10.0 * u.TeV,
            "optimization_energy_min": 0.1 * u.TeV,
            "optimization_energy_max": 10.0 * u.TeV,
            "reduced_core_radius": None,
            "reduced_view_cone_radius": None,
            "plot_diagnostics": True,
            "output_file": str(tmp_path / "estimate.ecsv"),
        }
    )

    mock_plot.assert_called_once()
    assert mock_plot.call_args.args[2] == {
        "zenith": 20.0 * u.deg,
        "azimuth": 180.0 * u.deg,
        "nsb_level": 1.0,
    }


def test_estimator_reports_positive_required_events(mocker, tmp_path):
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
        "target_triggered_events": None,
        "br_energy_min": 0.1 * u.TeV,
        "br_energy_max": 10.0 * u.TeV,
        "optimization_energy_min": 0.1 * u.TeV,
        "optimization_energy_max": 10.0 * u.TeV,
        "reduced_core_radius": None,
        "reduced_view_cone_radius": None,
        "output_file": str(tmp_path / "estimate.ecsv"),
    }

    result = monte_carlo_statistics_estimator.estimate_monte_carlo_statistics(args)

    assert result["array_name"][0] == "alpha"
    assert result["primary_particle"][0] == "gamma"
    assert result["zenith"].quantity[0].to_value(u.deg) == pytest.approx(20.0)
    assert result["azimuth"].quantity[0].to_value(u.deg) == pytest.approx(180.0)
    assert result["nsb_level"][0] == pytest.approx(1.0)
    assert result["estimated_total_events"][0] > 0.0
    assert float(result["estimated_total_events"][0]).is_integer()
    assert "limiting_energy_low" not in result.colnames
    assert "limiting_energy_high" not in result.colnames
    assert "limiting_angular_distance_low" not in result.colnames
    assert "limiting_angular_distance_high" not in result.colnames
    assert "limiting_expected_trigger_count" not in result.colnames
    assert "limiting_trigger_efficiency" not in result.colnames
    assert "optimization_bins_used" not in result.colnames
    assert "optimization_bins_skipped" not in result.colnames
    assert "original_core_scatter_radius" not in result.colnames
    assert "original_view_cone_radius" not in result.colnames


def test_estimator_logs_reference_validation_summary(mocker, tmp_path, caplog):
    metadata, bins = _build_reference_tables()
    mocker.patch(
        _LOAD_HISTOGRAMS,
        return_value=(metadata, bins),
    )
    caplog.set_level("INFO")

    monte_carlo_statistics_estimator.estimate_monte_carlo_statistics(
        {
            "input": "unused.hdf5",
            "array_names": None,
            "spectral_index": -2.0,
            "target_relative_uncertainty": 0.1,
            "target_triggered_events": None,
            "optimization_energy_min": 0.1 * u.TeV,
            "optimization_energy_max": 10.0 * u.TeV,
            "reduced_core_radius": None,
            "reduced_view_cone_radius": None,
            "output_file": str(tmp_path / "estimate.ecsv"),
        }
    )

    assert (
        "Using trigger histogram for array_layout=alpha "
        "(zenith=20.000 deg, azimuth=180.000 deg, nsb_level=1.0): "
        "simulated_events=200 triggered_events=75 overall_trigger_efficiency=0.375"
    ) in caplog.text


def test_estimator_logs_overall_trigger_probability_for_target_triggered_events(
    mocker, tmp_path, caplog
):
    metadata, bins = _build_reference_tables()
    mocker.patch(
        _LOAD_HISTOGRAMS,
        return_value=(metadata, bins),
    )
    caplog.set_level("INFO")

    monte_carlo_statistics_estimator.estimate_monte_carlo_statistics(
        {
            "input": "unused.hdf5",
            "array_names": None,
            "spectral_index": -2.0,
            "target_relative_uncertainty": None,
            "target_triggered_events": 25,
            "optimization_energy_min": 0.1 * u.TeV,
            "optimization_energy_max": 10.0 * u.TeV,
            "reduced_core_radius": None,
            "reduced_view_cone_radius": None,
            "output_file": str(tmp_path / "estimate.ecsv"),
        }
    )

    assert (
        "Using user-provided target spectral index -2 for array_layout=alpha "
        "(source spectral index -2 from trigger histogram metadata)."
    ) in caplog.text
    assert (
        "Overall trigger probability in selected optimization range for array_layout=alpha "
        "(zenith=20.000 deg, azimuth=180.000 deg, nsb_level=1.0): 0.375"
    ) in caplog.text


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
                "target_triggered_events": None,
                "optimization_energy_min": 0.1 * u.TeV,
                "optimization_energy_max": 1.0 * u.TeV,
                "reduced_core_radius": 50.0 * u.m,
                "reduced_view_cone_radius": None,
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
        "target_triggered_events": None,
        "br_energy_min": None,
        "br_energy_max": None,
        "optimization_energy_min": None,
        "optimization_energy_max": None,
        "reduced_core_radius": None,
        "reduced_view_cone_radius": None,
        "output_file": str(tmp_path / "estimate.ecsv"),
    }

    result = monte_carlo_statistics_estimator.estimate_monte_carlo_statistics(args)

    assert "effective_core_scatter_radius" not in result.colnames
    assert "original_core_scatter_radius" not in result.colnames
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
            "target_triggered_events": None,
            "br_energy_min": None,
            "br_energy_max": None,
            "optimization_energy_min": None,
            "optimization_energy_max": None,
            "reduced_core_radius": None,
            "reduced_view_cone_radius": None,
            "output_file": str(tmp_path / "estimate.ecsv"),
        }
    )

    assert result["array_name"][0] == "CTAO-North-Alpha"


def test_estimator_view_cone_override_changes_required_events(mocker, tmp_path):
    metadata, bins = _build_reference_tables_with_multiple_angular_bins()
    mocker.patch(
        _LOAD_HISTOGRAMS,
        return_value=(metadata, bins),
    )

    common_args = {
        "input": "unused.hdf5",
        "array_names": None,
        "spectral_index": -2.0,
        "target_relative_uncertainty": 0.1,
        "target_triggered_events": None,
        "optimization_energy_min": 0.1 * u.TeV,
        "optimization_energy_max": 10.0 * u.TeV,
        "reduced_core_radius": None,
    }

    original = monte_carlo_statistics_estimator.estimate_monte_carlo_statistics(
        common_args
        | {
            "reduced_view_cone_radius": None,
            "output_file": str(tmp_path / "full.ecsv"),
        }
    )
    reduced = monte_carlo_statistics_estimator.estimate_monte_carlo_statistics(
        common_args
        | {
            "reduced_view_cone_radius": 5.0 * u.deg,
            "output_file": str(tmp_path / "reduced.ecsv"),
        }
    )

    assert original["estimated_total_events"][0] != pytest.approx(
        reduced["estimated_total_events"][0]
    )
    assert reduced.meta["reduced_view_cone_radius"].to_value(u.deg) == pytest.approx(5.0)
    assert "effective_view_cone_radius" not in reduced.colnames


def test_estimator_supports_target_triggered_events(mocker, tmp_path):
    metadata, bins = _build_reference_tables()
    mocker.patch(
        _LOAD_HISTOGRAMS,
        return_value=(metadata, bins),
    )

    result = monte_carlo_statistics_estimator.estimate_monte_carlo_statistics(
        {
            "input": "unused.hdf5",
            "array_names": None,
            "spectral_index": -2.0,
            "target_relative_uncertainty": None,
            "target_triggered_events": 25,
            "optimization_energy_min": 0.1 * u.TeV,
            "optimization_energy_max": 10.0 * u.TeV,
            "reduced_core_radius": None,
            "reduced_view_cone_radius": None,
            "output_file": str(tmp_path / "estimate.ecsv"),
        }
    )

    assert result.meta["target_triggered_events"] == 25
    assert result.meta["spectral_index"] == pytest.approx(-2.0)
    assert result.meta["optimization_energy_min"].to_value(u.TeV) == pytest.approx(0.1)
    assert result.meta["optimization_energy_max"].to_value(u.TeV) == pytest.approx(10.0)
    assert "target_triggered_events" not in result.colnames
    assert "spectral_index" not in result.colnames
    assert "optimization_energy_min" not in result.colnames
    assert "optimization_energy_max" not in result.colnames
    assert "original_core_scatter_radius" not in result.colnames
    assert "original_view_cone_radius" not in result.colnames
    assert result["estimated_total_events"][0] == 67

    written = Table.read(tmp_path / "estimate.ecsv", format="ascii.ecsv")
    assert written.meta["target_triggered_events"] == 25
    assert written.meta["spectral_index"] == pytest.approx(-2.0)
    assert written.meta["optimization_energy_min"].to_value(u.TeV) == pytest.approx(0.1)
    assert written.meta["optimization_energy_max"].to_value(u.TeV) == pytest.approx(10.0)


def test_estimator_writes_complete_shared_configuration_to_metadata(mocker, tmp_path):
    metadata, bins = _build_reference_tables()
    mocker.patch(
        _LOAD_HISTOGRAMS,
        return_value=(metadata, bins),
    )

    result = monte_carlo_statistics_estimator.estimate_monte_carlo_statistics(
        {
            "input": "unused.hdf5",
            "array_names": None,
            "spectral_index": -2.0,
            "target_relative_uncertainty": 0.1,
            "target_triggered_events": None,
            "optimization_energy_min": 0.1 * u.TeV,
            "optimization_energy_max": 10.0 * u.TeV,
            "reduced_core_radius": 50.0 * u.m,
            "reduced_view_cone_radius": 5.0 * u.deg,
            "output_file": str(tmp_path / "estimate.ecsv"),
        }
    )

    assert result.meta["spectral_index"] == pytest.approx(-2.0)
    assert result.meta["target_relative_uncertainty"] == pytest.approx(0.1)
    assert result.meta["optimization_energy_min"].to_value(u.TeV) == pytest.approx(0.1)
    assert result.meta["optimization_energy_max"].to_value(u.TeV) == pytest.approx(10.0)
    assert result.meta["reduced_core_radius"].to_value(u.m) == pytest.approx(50.0)
    assert result.meta["reduced_view_cone_radius"].to_value(u.deg) == pytest.approx(5.0)
    assert "effective_core_scatter_radius" not in result.colnames
    assert "effective_view_cone_radius" not in result.colnames
    assert "original_core_scatter_radius" not in result.colnames
    assert "original_view_cone_radius" not in result.colnames


def test_estimator_uses_histogram_spectral_index_when_no_override_is_given(mocker, tmp_path):
    metadata, bins = _build_reference_tables()
    mocker.patch(
        _LOAD_HISTOGRAMS,
        return_value=(metadata, bins),
    )

    result = monte_carlo_statistics_estimator.estimate_monte_carlo_statistics(
        {
            "input": "unused.hdf5",
            "array_names": None,
            "spectral_index": None,
            "target_relative_uncertainty": None,
            "target_triggered_events": 25,
            "optimization_energy_min": 0.1 * u.TeV,
            "optimization_energy_max": 10.0 * u.TeV,
            "reduced_core_radius": None,
            "reduced_view_cone_radius": None,
            "output_file": str(tmp_path / "estimate.ecsv"),
        }
    )

    assert result.meta["spectral_index"] == pytest.approx(-2.0)
    assert result["estimated_total_events"][0] == 67


def test_estimator_warns_and_falls_back_when_histogram_spectral_index_is_missing(
    mocker, tmp_path, caplog
):
    metadata, bins = _build_reference_tables()
    metadata.remove_column("spectral_index")
    mocker.patch(
        _LOAD_HISTOGRAMS,
        return_value=(metadata, bins),
    )
    caplog.set_level("WARNING")

    result = monte_carlo_statistics_estimator.estimate_monte_carlo_statistics(
        {
            "input": "unused.hdf5",
            "array_names": None,
            "spectral_index": None,
            "target_relative_uncertainty": None,
            "target_triggered_events": 25,
            "optimization_energy_min": 0.1 * u.TeV,
            "optimization_energy_max": 10.0 * u.TeV,
            "reduced_core_radius": None,
            "reduced_view_cone_radius": None,
            "output_file": str(tmp_path / "estimate.ecsv"),
        }
    )

    assert "No spectral index found in trigger histogram metadata" in caplog.text
    assert result.meta["spectral_index"] == pytest.approx(-2.0)
    assert result["estimated_total_events"][0] == 67


def test_estimator_logs_when_using_histogram_spectral_index(mocker, tmp_path, caplog):
    metadata, bins = _build_reference_tables()
    mocker.patch(
        _LOAD_HISTOGRAMS,
        return_value=(metadata, bins),
    )
    caplog.set_level("INFO")

    monte_carlo_statistics_estimator.estimate_monte_carlo_statistics(
        {
            "input": "unused.hdf5",
            "array_names": None,
            "spectral_index": None,
            "target_relative_uncertainty": None,
            "target_triggered_events": 25,
            "optimization_energy_min": 0.1 * u.TeV,
            "optimization_energy_max": 10.0 * u.TeV,
            "reduced_core_radius": None,
            "reduced_view_cone_radius": None,
            "output_file": str(tmp_path / "estimate.ecsv"),
        }
    )

    assert "Using spectral index -2 from trigger histogram metadata for array_layout=alpha." in (
        caplog.text
    )
