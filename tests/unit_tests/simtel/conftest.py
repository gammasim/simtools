"""Shared physical setup for calibration simulation and format-writer tests."""

from unittest.mock import Mock

import pytest

from simtools.simtel.light_emission_config_writer import LightEmissionConfigWriter
from simtools.simtel.simulator_light_emission import SimulatorLightEmission


@pytest.fixture
def simulator_instance():
    """Create a fresh mock SimulatorLightEmission instance for each test."""
    inst = SimulatorLightEmission.__new__(SimulatorLightEmission)
    # Create fresh mocks for each test to avoid cross-test contamination
    inst.calibration_model = Mock()
    inst.telescope_model = Mock()
    inst.site_model = Mock()
    inst.light_emission_config = {}
    inst.submission_files = Mock()
    inst.output_directory = "/test/output"
    inst._logger = Mock()
    inst.runner_service = Mock()
    inst.io_handler = Mock()
    writer = LightEmissionConfigWriter(inst)
    writer._logger = inst._logger
    inst.configuration_writers = {"light_emission": writer}
    return inst
