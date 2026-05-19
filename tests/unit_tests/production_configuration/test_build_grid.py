from unittest.mock import Mock, patch

import astropy.units as u
import pytest

from simtools.production_configuration.build_grid import (
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
        "array_layout_name": "layout",
        "corsika_le_interaction": "urqmd",
        "corsika_he_interaction": "epos",
        "lookup_table": "limits.ecsv",
        "telescope_ids": ["MSTN-15"],
        "simtel_file": None,
    }

    rows = build_simulation_jobs(args_dict)

    assert len(rows) == 2
    assert rows[0]["azimuth_angle"] == 180 * u.deg
    assert rows[0]["zenith_angle"] == 30 * u.deg
    assert rows[0]["energy_min"] == 50 * u.GeV
    assert rows[0]["energy_max"] == 100 * u.GeV
    assert rows[0]["core_scatter_max"] == 250 * u.m
    assert rows[0]["view_cone_max"] == 3 * u.deg
    assert rows[0]["run_number"] == 11
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
        "telescope_ids": None,
        "simtel_file": None,
    }

    rows = build_simulation_jobs(args_dict)

    assert rows[0]["view_cone_max"].to_value(u.deg) == pytest.approx(2.5)
