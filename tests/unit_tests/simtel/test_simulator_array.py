#!/usr/bin/python3

import logging
import re
from pathlib import Path

import pytest

from simtools.runners.simtel_runner import InvalidOutputFileError
from simtools.simtel.simulator_array import SimulatorArray

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture()
def simtel_runner(corsika_config, simtel_path):
    return SimulatorArray(
        corsika_config=corsika_config,
        simtel_path=simtel_path,
        label="test-simtel-runner",
        keep_seeds=False,
        use_multipipe=False,
    )


def test_simtel_runner(simtel_runner):
    sr = simtel_runner
    assert "simtel" in str(sr._directory["output"])
    assert "simtel" in str(sr._directory["data"])
    assert isinstance(sr._directory["data"], Path)


def test_make_run_command(simtel_runner):
    run_command = simtel_runner._make_run_command(
        run_number=3, input_file="test_make_run_command.inp"
    )
    assert "sim_telarray" in run_command
    assert "-run" in run_command
    assert "3" in run_command
    assert "test-simtel-runner.zst" in run_command
    assert "test_make_run_command.inp" in run_command


def test_check_run_result(simtel_runner):
    expected_pattern = r"sim_telarray output file .+ does not exist\."
    with pytest.raises(InvalidOutputFileError, match=expected_pattern):
        assert simtel_runner._check_run_result(run_number=3)


def test_get_resources(simtel_runner):
    with pytest.raises(FileNotFoundError):
        simtel_runner.get_resources()


def test_get_file_name(simtel_runner):
    with pytest.raises(
        ValueError, match=re.escape("simulation_software (test) is not supported in SimulatorArray")
    ):
        simtel_runner.get_file_name(simulation_software="test")

    assert isinstance(
        simtel_runner.get_file_name(simulation_software="simtel", file_type="output", run_number=3),
        Path,
    )
