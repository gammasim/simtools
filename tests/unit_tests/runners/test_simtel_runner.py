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
    with pytest.raises(SimtelExecutionError, match=r"Simtel log file does not exist."):
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


def test_raise_simtel_error_with_log_file(simtel_runner, tmp_path, mocker):
    log_file = tmp_path / "test.log"
    log_file.write_text("Line 1\nLine 2\nLine 3\nError occurred\n")
    simtel_runner._log_file = log_file

    mocker.patch("simtools.utils.general.get_log_excerpt", return_value="Error occurred")

    with pytest.raises(SimtelExecutionError, match=r"Error occurred"):
        simtel_runner._raise_simtel_error()


def test_raise_simtel_error_without_log_file(simtel_runner):
    if hasattr(simtel_runner, "_log_file"):
        delattr(simtel_runner, "_log_file")

    with pytest.raises(SimtelExecutionError, match=r"Simtel log file does not exist."):
        simtel_runner._raise_simtel_error()


def test_check_run_result(simtel_runner):
    simtel_runner._check_run_result(run_number=5)


def test_run_with_runs_per_set(simtel_runner, mocker):
    mock_make_run_command = mocker.patch.object(
        simtel_runner, "_make_run_command", return_value=("echo test", None, None)
    )
    mock_run_simtel = mocker.patch.object(
        simtel_runner, "_run_simtel_and_check_output", return_value=0
    )
    mock_check_result = mocker.patch.object(simtel_runner, "_check_run_result")

    simtel_runner.runs_per_set = 3
    simtel_runner.run(test=False, input_file="test_input", run_number=10)

    mock_make_run_command.assert_called_once_with(run_number=10, input_file="test_input")
    assert mock_run_simtel.call_count == 3
    mock_check_result.assert_called_once_with(run_number=10)


def test_run_with_test_mode(simtel_runner, mocker):
    mock_make_run_command = mocker.patch.object(
        simtel_runner, "_make_run_command", return_value=("echo test", None, None)
    )
    mock_run_simtel = mocker.patch.object(
        simtel_runner, "_run_simtel_and_check_output", return_value=0
    )
    mock_check_result = mocker.patch.object(simtel_runner, "_check_run_result")

    simtel_runner.run(test=True, input_file="test_input", run_number=7)

    mock_make_run_command.assert_called_once_with(run_number=7, input_file="test_input")
    mock_run_simtel.assert_called_once_with("echo test", None, None)
    mock_check_result.assert_called_once_with(run_number=7)


def test_run_calls_check_run_result(simtel_runner, mocker):
    mocker.patch.object(simtel_runner, "_make_run_command", return_value=("echo test", None, None))
    mocker.patch.object(simtel_runner, "_run_simtel_and_check_output", return_value=0)
    mock_check_result = mocker.patch.object(simtel_runner, "_check_run_result")

    simtel_runner.run(test=True, input_file="test", run_number=15)

    mock_check_result.assert_called_once_with(run_number=15)
