#!/usr/bin/python3

import logging

import pytest

from simtools.runners.simtel_runner import SimtelRunner

logger = logging.getLogger()


@pytest.fixture
def simtel_runner(corsika_config_mock_array_model):
    return SimtelRunner(label="test", corsika_config=corsika_config_mock_array_model)


def test_run(simtel_runner, caplog, mocker):
    with pytest.raises(NotImplementedError, match=r"Must be implemented in concrete subclass"):
        simtel_runner.run(test=True, input_file="test", run_number=5)


def test_run_with_runs_per_set(simtel_runner, mocker):
    mock_make_run_command = mocker.patch.object(
        simtel_runner, "make_run_command", return_value=("echo test", None, None)
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
        simtel_runner, "make_run_command", return_value=("echo test", None, None)
    )
    mock_job_manager_submit = mocker.patch(
        "simtools.job_execution.job_manager.submit", return_value=0
    )

    simtel_runner.run(test=True, input_file="test_input", run_number=7)

    mock_make_run_command.assert_called_once_with(run_number=7, input_file="test_input")
    mock_job_manager_submit.assert_called_once_with(
        "echo test", out_file=None, err_file=None, env={"SIM_TELARRAY_CONFIG_PATH": ""}
    )


def test_run_completes_successfully(simtel_runner, mocker):
    mocker.patch.object(simtel_runner, "make_run_command", return_value=("echo test", None, None))
    mocker.patch("simtools.job_execution.job_manager.submit", return_value=0)

    # Test that run completes without error when properly mocked
    result = simtel_runner.run(test=True, input_file="test", run_number=15)
    # run method doesn't return anything, so just check it completed
    assert result is None
