import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table

from simtools.production_configuration import trigger_statistics_estimator


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
                "angular_bin_index": 0,
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
                "angular_bin_index": 0,
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
    assert trigger_statistics_estimator._resolve_effective_throw_radius(100.0 * u.m).to_value(
        u.m
    ) == pytest.approx(100.0)
    assert trigger_statistics_estimator._resolve_effective_throw_radius(
        100.0 * u.m, 80.0 * u.m
    ).to_value(u.m) == pytest.approx(80.0)


def test_resolve_effective_throw_radius_rejects_invalid_override():
    with pytest.raises(ValueError, match="positive"):
        trigger_statistics_estimator._resolve_effective_throw_radius(100.0 * u.m, 0.0 * u.m)
    with pytest.raises(ValueError, match="cannot exceed"):
        trigger_statistics_estimator._resolve_effective_throw_radius(100.0 * u.m, 120.0 * u.m)


def test_compute_effective_area_matrix_scales_with_radius():
    matrix = np.array([[0.5, 0.25]])
    original = trigger_statistics_estimator._compute_effective_area_matrix(matrix, 100.0 * u.m)
    reduced = trigger_statistics_estimator._compute_effective_area_matrix(matrix, 50.0 * u.m)

    np.testing.assert_allclose(reduced, original / 4.0)


def test_estimator_radius_override_changes_geometric_normalization_not_required_events(
    mocker, tmp_path
):
    metadata, bins = _build_reference_tables()
    mocker.patch(
        (
            "simtools.production_configuration.trigger_statistics_estimator."
            "load_trigger_statistics_reference"
        ),
        return_value=(metadata, bins),
    )

    common_args = {
        "input": "unused.hdf5",
        "reference_ids": None,
        "production_indices": None,
        "array_names": None,
        "spectral_index": -2.0,
        "target_relative_uncertainty": 0.1,
        "thrown_energy_min": 0.1 * u.TeV,
        "thrown_energy_max": 10.0 * u.TeV,
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

    original = trigger_statistics_estimator.estimate_trigger_statistics(args_original)
    reduced = trigger_statistics_estimator.estimate_trigger_statistics(args_reduced)

    assert original["required_total_thrown_events"][0] == pytest.approx(
        reduced["required_total_thrown_events"][0]
    )
    assert reduced["effective_core_scatter_radius"][0].to_value(u.m) == pytest.approx(50.0)
    assert reduced["effective_scatter_area"][0].to_value(u.m**2) == pytest.approx(
        original["effective_scatter_area"][0].to_value(u.m**2) / 4.0
    )
    assert reduced["limiting_effective_area"][0].to_value(u.m**2) == pytest.approx(
        original["limiting_effective_area"][0].to_value(u.m**2) / 4.0
    )


def test_estimator_reports_limiting_bin_and_positive_required_events(mocker, tmp_path):
    metadata, bins = _build_reference_tables()
    mocker.patch(
        (
            "simtools.production_configuration.trigger_statistics_estimator."
            "load_trigger_statistics_reference"
        ),
        return_value=(metadata, bins),
    )

    args = {
        "input": "unused.hdf5",
        "reference_ids": None,
        "production_indices": None,
        "array_names": None,
        "spectral_index": -2.0,
        "target_relative_uncertainty": 0.1,
        "thrown_energy_min": 0.1 * u.TeV,
        "thrown_energy_max": 10.0 * u.TeV,
        "optimization_energy_min": 0.1 * u.TeV,
        "optimization_energy_max": 10.0 * u.TeV,
        "reduced_core_radius": None,
        "output_file": str(tmp_path / "estimate.ecsv"),
    }

    result = trigger_statistics_estimator.estimate_trigger_statistics(args)

    assert result["reference_id"][0] == "ref-1"
    assert result["required_total_thrown_events"][0] > 0.0
    assert result["limiting_energy_low"][0] in (0.1 * u.TeV, 1.0 * u.TeV)
