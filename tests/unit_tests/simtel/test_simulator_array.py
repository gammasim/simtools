#!/usr/bin/python3

import logging

import pytest

from simtools.simtel.simulator_array import SimulatorArray

logger = logging.getLogger()


@pytest.fixture
def simtel_runner(corsika_config_mock_array_model):
    return SimulatorArray(
        corsika_config=corsika_config_mock_array_model,
        label="test-simtel-runner",
    )
