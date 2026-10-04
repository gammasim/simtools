"""Test common simulation settings without CORSIKA configuration writing."""

from types import SimpleNamespace

import astropy.units as u
import pytest

from simtools.corsika.primary_particle import PrimaryParticle
from simtools.simulation.parameters import CALIBRATION_RUN_MODES, SimulationParameters


def test_physical_settings_from_arguments(mocker):
    parameters = SimulationParameters.from_args(
        {
            "primary": "proton",
            "primary_id_type": "common_name",
            "zenith_angle": 30 * u.deg,
            "azimuth_angle": 120 * u.deg,
            "showers_per_run": 7,
            "core_scatter": [3, 200 * u.m],
            "view_cone": [0 * u.deg, 5 * u.deg],
            "curved_atmosphere_min_zenith_angle": 25 * u.deg,
        },
        42,
        mocker.Mock(),
    )
    assert parameters.primary_particle.name == "proton"
    assert parameters.run_number == 42
    assert parameters.shower_events == 7
    assert parameters.mc_events == 21
    assert parameters.zenith_angle == pytest.approx(30)
    assert parameters.azimuth_angle == pytest.approx(120)
    assert parameters.viewcone_max == pytest.approx(5)
    assert parameters.use_curved_atmosphere
    assert not parameters.is_calibration_run()


@pytest.mark.parametrize("run_mode", sorted(CALIBRATION_RUN_MODES))
def test_calibration_requires_no_shower_writer(run_mode, mocker):
    parameters = SimulationParameters.from_args({"run_mode": run_mode}, 1, mocker.Mock())
    assert parameters.is_calibration_run()
    assert parameters.shower_events == parameters.mc_events == 1
    assert parameters.viewcone_max == 0


def test_defaults(mocker):
    parameters = SimulationParameters.from_args({}, 1, mocker.Mock())
    assert parameters.zenith_angle == 20
    assert parameters.azimuth_angle == 0
    assert parameters.shower_events == parameters.mc_events == 0
    assert not parameters.use_curved_atmosphere


@pytest.mark.parametrize("height", [None, 2200, 1800])
def test_input_file_uses_selected_reader(height, mocker):
    metadata = {
        "primary_particle": PrimaryParticle("common_name", "gamma"),
        "zenith_angle": 42,
        "azimuth_angle": 180,
        "zenith_min": 40,
        "shower_events": 3,
        "mc_events": 6,
        "viewcone_max": 5,
        "observation_levels": [] if height is None else [height],
    }
    reader = SimpleNamespace(read_simulation_parameters=lambda: metadata.copy())
    factory = mocker.patch("simtools.simulation.parameters.get_reader", return_value=reader)
    site = mocker.Mock()
    site.get_parameter_value_with_unit.return_value = 2200 * u.m
    args = {"corsika_file": "showers.new", "simulation_file_format": "example"}
    if height == 1800:
        with pytest.raises(ValueError, match="altitude does not match"):
            SimulationParameters.from_args(args, 1, site)
    else:
        parameters = SimulationParameters.from_args(args, 1, site)
        assert parameters.zenith_angle == 42
        assert parameters.mc_events == 6
        assert parameters.primary_particle.name == "gamma"
    factory.assert_called_once_with("showers.new", "example")
