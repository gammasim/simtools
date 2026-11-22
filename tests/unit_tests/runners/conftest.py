"""Fixtures of the runner tests."""

from pathlib import Path
from unittest.mock import Mock

import astropy.units as u
import pytest

from simtools.simtel.light_emission_config_writer import LightEmissionConfigWriter
from simtools.simtel.simulator_light_emission import SimulatorLightEmission


@pytest.fixture
def light_simulation(run_context, tmp_test_directory, monkeypatch):
    """Light-emission simulation of a flasher at one telescope, with mocked models."""
    simulation = SimulatorLightEmission.__new__(SimulatorLightEmission)
    simulation.context = run_context
    simulation.telescope = "MSTN-04"
    simulation.label = "test"
    simulation._logger = Mock()
    simulation._seed_pair = (11, 13)
    simulation.telescope_model = Mock(
        config_file_directory="/mock/config",
        config_file_path="/mock/config/telescope.cfg",
        model_version="6.0.2",
    )
    simulation.telescope_model.get_parameter_value.return_value = 0  # single mirror
    simulation.site_model = Mock(site="North")
    simulation.site_model.get_parameter_value_with_unit.return_value = 2156 * u.m
    simulation.site_model.get_parameter_value.return_value = "atm_trans.dat"
    simulation.calibration_model = Mock()
    simulation.calibration_model.name = "MSFx-FlashCam"
    simulation.light_emission_config = {
        "light_source_type": "flat_fielding",
        "run_number": 3,
        "max_photoelectrons": 1000,
    }
    simulation.runner_service = Mock()
    simulation.runner_service.get_file_name.side_effect = lambda file_type, run_number=None: (
        Path(tmp_test_directory) / f"{file_type}.dat"
    )
    simulation.runner_service.get_stage_file_name.side_effect = (
        lambda stage, file_type, run_number=None: (
            Path(tmp_test_directory) / f"run.{stage}.{file_type}"
        )
    )
    simulation.submission_files = Mock()
    simulation.submission_files.get_file_name.side_effect = lambda file_type: (
        Path(tmp_test_directory) / "sub_script.sh"
    )
    writer = LightEmissionConfigWriter(simulation)
    simulation.configuration_writers = {"light_emission": writer}
    monkeypatch.setattr(writer, "make_command", lambda iact_output: f"ff-1m -o {iact_output}")
    return simulation
