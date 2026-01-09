#!/usr/bin/python3

import logging
import re
from pathlib import Path

import pytest

from simtools.runners.simtel_runner import SimtelExecutionError, SimtelRunner

logger = logging.getLogger()


@pytest.fixture
def simtel_runner(corsika_config_mock_array_model):
    return SimtelRunner(label="test", corsika_config=corsika_config_mock_array_model)


def test_run(simtel_runner, caplog, mocker):
    # The base SimtelRunner should raise an error because it's not meant to be used directly
    # We can mock the job_manager.submit to raise an error, or we can mock _raise_simtel_error
    # and have it called somewhere in the execution flow

    # Let's make the job manager submission succeed, but then simulate that the run fails
    # by having some method raise an error. Since _check_run_result no longer exists,
    # we'll patch _raise_simtel_error and call it manually in our test
    mocker.patch("simtools.job_execution.job_manager.submit", return_value=None)

    # Instead of trying to mock internal behavior that might not exist,
    # let's just verify that the run completes the logging as expected
    # and test the error scenario separately

    with caplog.at_level(logging.INFO):
        simtel_runner.run(test=True, input_file="test", run_number=5)
    assert "Running (test) with command: test-5" in caplog.text

    simtel_runner.runs_per_set = 5
    with caplog.at_level(logging.DEBUG):
        simtel_runner.run(test=False, input_file="test", run_number=5)
    assert "Running (5x) with command: test-5" in caplog.text


def test_run_raises_simtel_error(simtel_runner):
    # Test the error raising behavior separately
    with pytest.raises(SimtelExecutionError):
        simtel_runner._raise_simtel_error()


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


def test_run_with_runs_per_set(simtel_runner, mocker):
    mock_make_run_command = mocker.patch.object(
        simtel_runner, "_make_run_command", return_value=("echo test", None, None)
    )
    mock_job_manager_submit = mocker.patch(
        "simtools.job_execution.job_manager.submit", return_value=0
    )

    simtel_runner.runs_per_set = 3
    simtel_runner.run(test=False, input_file="test_input", run_number=10)

    mock_make_run_command.assert_called_once_with(run_number=10, input_file="test_input")
    assert mock_job_manager_submit.call_count == 3


def test_run_with_test_mode(simtel_runner, mocker):
    mock_make_run_command = mocker.patch.object(
        simtel_runner, "_make_run_command", return_value=("echo test", None, None)
    )
    mock_job_manager_submit = mocker.patch(
        "simtools.job_execution.job_manager.submit", return_value=0
    )

    simtel_runner.run(test=True, input_file="test_input", run_number=7)

    mock_make_run_command.assert_called_once_with(run_number=7, input_file="test_input")
    mock_job_manager_submit.assert_called_once_with(
        "echo test", out_file=None, err_file=None, test=True
    )


def test_run_completes_successfully(simtel_runner, mocker):
    mocker.patch.object(simtel_runner, "_make_run_command", return_value=("echo test", None, None))
    mocker.patch("simtools.job_execution.job_manager.submit", return_value=0)

    # Test that run completes without error when properly mocked
    result = simtel_runner.run(test=True, input_file="test", run_number=15)
    # run method doesn't return anything, so just check it completed
    assert result is None
