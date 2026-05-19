from unittest.mock import Mock, patch

import astropy.units as u
import pytest

from simtools.production_configuration.build_grid import (
    build_job_grid_metadata,
    build_simulation_jobs,
    get_viewcone_max_for_zenith_angle,
    resolve_single_model_version,
)


def test_get_viewcone_max_for_zenith_angle_without_lookup_table():
    max_view_cone = get_viewcone_max_for_zenith_angle(20 * u.deg, [0 * u.deg, 10 * u.deg], None)

    assert max_view_cone == 10 * u.deg


def test_resolve_single_model_version_uses_first_list_entry():
    assert resolve_single_model_version(["7.0.0", "7.1.0"]) == "7.0.0"
    assert resolve_single_model_version("7.0.0") == "7.0.0"


def test_build_job_grid_metadata_includes_job_context():
    metadata = build_job_grid_metadata(
        {
            "site": "North",
            "simulation_software": "corsika_sim_telarray",
            "coordinate_system": "ra_dec",
            "observing_time": "2017-09-16 00:00:00",
            "lookup_table": "limits.ecsv",
        }
    )

    assert metadata["site"] == "North"
    assert metadata["simulation_software"] == "corsika_sim_telarray"
    assert metadata["coordinate_system"] == "ra_dec"
    assert metadata["observing_time_utc"].startswith("2017-09-16T00:00:00")
    assert metadata["lookup_table"] == "limits.ecsv"


@patch("simtools.production_configuration.build_grid.build_production_grid_engine")
def test_build_simulation_jobs_uses_shared_axis_defined_grid(mock_build_production_grid_engine):
    mock_grid_engine = Mock()
    mock_grid_engine.generate_simulation_grid.return_value = [
        {
            "ra": 12 * u.deg,
            "dec": -20 * u.deg,
            "azimuth": 180 * u.deg,
            "zenith_angle": 30 * u.deg,
            "lower_energy_threshold": 50 * u.GeV,
            "scatter_radius": 250 * u.m,
            "viewcone_radius": 3 * u.deg,
        }
    ]
    mock_build_production_grid_engine.return_value = mock_grid_engine
    args_dict = {
        "axes": "grid.yml",
        "coordinate_system": "ra_dec",
        "site": "North",
        "model_version": ["7.0.0"],
        "primary": ["gamma"],
        "azimuth_angle": 0 * u.deg,
        "zenith_angle": 20 * u.deg,
        "energy_range": [[30 * u.GeV, 100 * u.GeV]],
        "core_scatter": [10, 500 * u.m],
        "view_cone": [0 * u.deg, 5 * u.deg],
        "nshow": 5,
        "number_of_runs": 2,
        "run_number": 11,
        "array_layout_name": {"by_version": {"<7.0.0": "layout", ">=7.0.0": "layout-v2"}},
        "corsika_le_interaction": "urqmd",
        "corsika_he_interaction": "epos",
        "lookup_table": "limits.ecsv",
    }

    rows = build_simulation_jobs(args_dict)

    assert len(rows) == 2
    assert rows[0]["azimuth_angle"] == 180 * u.deg
    assert rows[0]["zenith_angle"] == 30 * u.deg
    assert rows[0]["energy_min"] == 50 * u.GeV
    assert rows[0]["energy_max"] == 100 * u.GeV
    assert rows[0]["core_scatter_number"] == 10
    assert rows[0]["core_scatter_max"] == 250 * u.m
    assert rows[0]["view_cone_min"] == 0 * u.deg
    assert rows[0]["view_cone_max"] == 3 * u.deg
    assert rows[0]["run_number"] == 11
    assert rows[0]["array_layout_name"] == "layout-v2"
    assert rows[1]["run_number"] == 12


@patch("simtools.production_configuration.build_grid.get_core_scatter_max_for_zenith_angle")
@patch("simtools.production_configuration.build_grid.get_viewcone_max_for_zenith_angle")
@patch("simtools.production_configuration.build_grid.get_energy_range_for_zenith_angle")
def test_build_simulation_jobs_adds_viewcone_limit_from_lookup(
    mock_get_energy_range_for_zenith_angle,
    mock_get_viewcone_max_for_zenith_angle,
    mock_get_core_scatter_max_for_zenith_angle,
):
    mock_get_energy_range_for_zenith_angle.return_value = (30 * u.GeV, 100 * u.GeV)
    mock_get_core_scatter_max_for_zenith_angle.return_value = 150 * u.m
    mock_get_viewcone_max_for_zenith_angle.return_value = 2.5 * u.deg
    args_dict = {
        "primary": ["gamma"],
        "azimuth_angle": [180 * u.deg],
        "zenith_angle": [30 * u.deg],
        "model_version": ["7.0.0"],
        "corsika_le_interaction": "urqmd",
        "corsika_he_interaction": "epos",
        "energy_range": [[30 * u.GeV, 100 * u.GeV]],
        "core_scatter": [10, 500 * u.m],
        "view_cone": [0 * u.deg, 5 * u.deg],
        "nshow": 5,
        "number_of_runs": 1,
        "run_number": 1,
        "corsika_limits": None,
        "array_layout_name": "layout",
    }

    rows = build_simulation_jobs(args_dict)

    assert rows[0]["core_scatter_number"] == 10
    assert rows[0]["view_cone_min"] == 0 * u.deg
    assert rows[0]["view_cone_max"].to_value(u.deg) == pytest.approx(2.5)


@patch("simtools.production_configuration.build_grid.get_viewcone_max_for_zenith_angle")
@patch("simtools.production_configuration.build_grid.get_core_scatter_max_for_zenith_angle")
@patch("simtools.production_configuration.build_grid.get_energy_range_for_zenith_angle")
@patch("simtools.production_configuration.build_grid.CorsikaLimitsLookup")
def test_build_simulation_jobs_builds_lookup_per_resolved_layout(
    mock_corsika_limits_lookup,
    mock_get_energy_range_for_zenith_angle,
    mock_get_core_scatter_max_for_zenith_angle,
    mock_get_viewcone_max_for_zenith_angle,
):
    args_dict = {
        "primary": ["gamma"],
        "azimuth_angle": [180 * u.deg],
        "zenith_angle": [30 * u.deg],
        "model_version": ["6.3.0", "7.0.0"],
        "corsika_le_interaction": "urqmd",
        "corsika_he_interaction": "epos",
        "energy_range": [[30 * u.GeV, 100 * u.GeV]],
        "core_scatter": [10, 500 * u.m],
        "view_cone": [0 * u.deg, 5 * u.deg],
        "nshow": 5,
        "number_of_runs": 1,
        "run_number": 1,
        "corsika_limits": "limits.ecsv",
        "array_layout_name": {"by_version": {"<7.0.0": "alpha", ">=7.0.0": "CTAO-North-Alpha"}},
    }
    mock_corsika_limits_lookup.return_value = Mock()
    mock_get_energy_range_for_zenith_angle.return_value = (30 * u.GeV, 100 * u.GeV)
    mock_get_core_scatter_max_for_zenith_angle.return_value = 100 * u.m
    mock_get_viewcone_max_for_zenith_angle.return_value = 2 * u.deg

    rows = build_simulation_jobs(args_dict)

    assert len(rows) == 2
    assert rows[0]["array_layout_name"] == "alpha"
    assert rows[1]["array_layout_name"] == "CTAO-North-Alpha"
    mock_corsika_limits_lookup.assert_any_call("limits.ecsv", array_layout_name="alpha")
    mock_corsika_limits_lookup.assert_any_call(
        "limits.ecsv",
        array_layout_name="CTAO-North-Alpha",
    )
