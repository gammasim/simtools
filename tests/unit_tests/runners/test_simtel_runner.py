#!/usr/bin/python3

import logging
import re
from pathlib import Path

import pytest

from simtools.runners.simtel_runner import SimtelExecutionError, SimtelRunner

logger = logging.getLogger()


@pytest.fixture
def simtel_runner(simtel_path, corsika_config_mock_array_model):
    return SimtelRunner(
        simtel_path=simtel_path, label="test", corsika_config=corsika_config_mock_array_model
    )


def test_repr(simtel_runner):
    assert repr(simtel_runner) == "SimtelRunner(label=test)\n"


def test_prepare_run_script(simtel_runner, tmp_test_directory):
    simtel_runner._base_directory = Path(tmp_test_directory)
    script_file = simtel_runner.prepare_run_script(
        test=True, input_file="test", run_number=1, extra_commands=None
    )
    assert script_file.exists()
    with open(script_file) as f:
        script_content = f.read()
        assert "/usr/bin/env bash" in script_content
        assert "RUNTIME" in script_content
        assert "test-1" in script_content

    simtel_runner.runs_per_set = 5
    script_file = simtel_runner.prepare_run_script(
        test=False, input_file="test", run_number=5, extra_commands="extra"
    )
    with open(script_file) as f:
        script_content = f.read()
        assert "test-5" in script_content
        assert script_content.count("test-5") == 5
        assert "extra" in script_content


def test_run(simtel_runner, caplog):
    with caplog.at_level(logging.INFO):
        with pytest.raises(SimtelExecutionError):
            simtel_runner.run(test=True, input_file="test", run_number=5)
    assert "Running (test) with command: test-5" in caplog.text
    simtel_runner.runs_per_set = 5
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(SimtelExecutionError):
            simtel_runner.run(test=False, input_file="test", run_number=5)
    assert "Running (5x) with command: test-5" in caplog.text


def test_run_simtel_and_check_output(simtel_runner):
    with pytest.raises(SimtelExecutionError):
        simtel_runner._run_simtel_and_check_output("test-5", None, None)
    assert simtel_runner._run_simtel_and_check_output("echo test", None, None) == 0


def test_make_run_command(simtel_runner, caplog):
    with caplog.at_level(logging.DEBUG):
        command, stdout_file, stderr_file = simtel_runner._make_run_command(
            input_file="test", run_number=5
        )
        assert command == "test-5"
        assert stdout_file is None
        assert stderr_file is None
    assert "make_run_command is being called from the base class" in caplog.text


def test_simtel_execution_error(simtel_runner):
    with pytest.raises(SimtelExecutionError):
        simtel_runner._raise_simtel_error()


def test_get_config_option(simtel_runner):
    assert simtel_runner.get_config_option("test", "value") == " -C test=value"
    assert simtel_runner.get_config_option("test", "value", weak_option=True) == " -W test=value"
    assert simtel_runner.get_config_option("test", "value", weak_option=False) == " -C test=value"
    assert simtel_runner.get_config_option("test") == " -C test"


def test_raise_simtel_error(simtel_runner):
    with pytest.raises(SimtelExecutionError, match="Simtel log file does not exist."):
        simtel_runner._raise_simtel_error()


def test_get_resources(simtel_runner):
    with pytest.raises(FileNotFoundError):
        simtel_runner.get_resources()


def test_get_file_name(simtel_runner):
    with pytest.raises(
        ValueError, match=re.escape("simulation_software (test) is not supported in SimulatorArray")
    ):
        simtel_runner.get_file_name(simulation_software="test")

    assert isinstance(
        simtel_runner.get_file_name(
            simulation_software="sim_telarray", file_type="output", run_number=3
        ),
        Path,
    )
