import logging
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

import simtools.job_execution.job_manager as jm
from simtools.job_execution.job_manager import JobExecutionError

LOG_EXCERPT = "log excerpt"
OS_SYSTEM = "os.system"
PATHLIB_PATH_EXISTS = "pathlib.Path.exists"

logger = logging.getLogger()


@pytest.fixture
def job_submitter():
    submitter = jm.JobManager()
    submitter._logger = MagicMock()
    submitter.test = True
    return submitter


@pytest.fixture
def output_log():
    """Fixture for the output log file."""
    return Path("output.log")


@pytest.fixture
def builtins_open():
    return "builtins.open"


@pytest.fixture
def subprocess_run():
    return "subprocess.run"


@pytest.fixture
def logfile_log():
    """Fixture for the general log file."""
    return Path("logfile.log")


@pytest.fixture
def script_file():
    """Fixture for the script file."""
    return Path("script.sh")


@pytest.fixture
def job_messages(script_file):
    """Fixture for the script message."""
    return {
        "script_message": f"Submitting script {script_file}",
        "job_output": "Job output stream output.out",
        "job_error_stream": "Job error stream output.err",
        "job_log_stream": "Job log stream output.job",
        "running_locally": "Running script locally",
        "log_excerpt": LOG_EXCERPT,
    }


@patch("simtools.utils.general")
def test_submit_local(
    mock_gen, job_submitter, mocker, output_log, logfile_log, script_file, job_messages
):
    mocker.patch(OS_SYSTEM, return_value=0)
    mock_gen.get_log_excerpt.return_value = LOG_EXCERPT
    mocker.patch(PATHLIB_PATH_EXISTS, return_value=False)

    job_submitter.submit(script_file, output_log, logfile_log)

    job_submitter._logger.info.assert_any_call(job_messages["script_message"])
    job_submitter._logger.info.assert_any_call(job_messages["job_output"])
    job_submitter._logger.info.assert_any_call(job_messages["job_error_stream"])
    job_submitter._logger.info.assert_any_call(job_messages["job_log_stream"])
    job_submitter._logger.info.assert_any_call(job_messages["running_locally"])
    job_submitter._logger.info.assert_any_call("Testing (local)")


@pytest.fixture
def job_submitter_real():
    submitter = jm.JobManager()
    submitter._logger = MagicMock()
    submitter.test = False
    return submitter


@patch("simtools.utils.general.get_log_excerpt")
@patch("simtools.utils.general.get_file_age")
def test_submit_local_real_failure(
    mock_get_file_age,
    mock_get_log_excerpt,
    job_submitter_real,
    mocker,
    output_log,
    logfile_log,
    script_file,
    job_messages,
    subprocess_run,
    builtins_open,
):
    mock_get_log_excerpt.return_value = job_messages["log_excerpt"]
    mock_get_file_age.return_value = 4
    mocker.patch(PATHLIB_PATH_EXISTS, return_value=True)

    # Mock file operations to prevent actual file creation
    mock_file = mocker.mock_open()
    mocker.patch(builtins_open, mock_file)

    # Mock subprocess.run to raise a CalledProcessError but also provide
    # a mock for stdout and stderr file handles
    mock_subprocess = mocker.patch(subprocess_run)
    mock_subprocess.side_effect = subprocess.CalledProcessError(1, str(script_file))

    with pytest.raises(JobExecutionError, match="See excerpt from log file above"):
        job_submitter_real.submit(script_file, output_log, logfile_log)

    job_submitter_real._logger.info.assert_any_call(job_messages["script_message"])
    job_submitter_real._logger.info.assert_any_call(job_messages["job_output"])
    job_submitter_real._logger.info.assert_any_call(job_messages["job_error_stream"])
    job_submitter_real._logger.info.assert_any_call(job_messages["job_log_stream"])
    job_submitter_real._logger.info.assert_any_call(job_messages["running_locally"])


@patch("simtools.utils.general")
def test_submit_local_success(
    mock_gen,
    job_submitter_real,
    mocker,
    output_log,
    logfile_log,
    script_file,
    job_messages,
    builtins_open,
    subprocess_run,
):
    mock_subprocess_run = mocker.patch(subprocess_run)
    mock_subprocess_run.return_value.returncode = 0
    mock_gen.get_log_excerpt.return_value = job_messages["log_excerpt"]
    mocker.patch(PATHLIB_PATH_EXISTS, return_value=False)

    with patch(builtins_open, mock_open(read_data="")):
        job_submitter_real.submit(script_file, output_log, logfile_log)

    job_submitter_real._logger.info.assert_any_call(job_messages["script_message"])
    job_submitter_real._logger.info.assert_any_call(job_messages["job_output"])
    job_submitter_real._logger.info.assert_any_call(job_messages["job_error_stream"])
    job_submitter_real._logger.info.assert_any_call(job_messages["job_log_stream"])
    job_submitter_real._logger.info.assert_any_call(job_messages["running_locally"])
    mock_subprocess_run.assert_called_with(
        f"{script_file}",
        shell=True,
        check=True,
        text=True,
        stdout=mocker.ANY,
        stderr=mocker.ANY,
    )

    mock_subprocess_run = mocker.patch(subprocess_run)
    mock_subprocess_run.return_value.returncode = 42
    with patch(builtins_open, mock_open(read_data="")):
        with pytest.raises(JobExecutionError, match="Job submission failed with return code 42"):
            job_submitter_real.submit(script_file, output_log, logfile_log)


@patch("simtools.utils.general")
def test_submit_local_test_mode(
    mock_gen,
    job_submitter,
    mocker,
    output_log,
    logfile_log,
    script_file,
    job_messages,
    subprocess_run,
):
    mock_subprocess_run = mocker.patch(subprocess_run)
    mock_subprocess_run.return_value.returncode = 0
    mock_gen.get_log_excerpt.return_value = job_messages["log_excerpt"]
    mocker.patch(PATHLIB_PATH_EXISTS, return_value=False)

    job_submitter.submit(script_file, output_log, logfile_log)

    job_submitter._logger.info.assert_any_call(job_messages["script_message"])
    job_submitter._logger.info.assert_any_call(job_messages["job_output"])
    job_submitter._logger.info.assert_any_call(job_messages["job_error_stream"])
    job_submitter._logger.info.assert_any_call(job_messages["job_log_stream"])
    job_submitter._logger.info.assert_any_call(job_messages["running_locally"])
    job_submitter._logger.info.assert_any_call("Testing (local)")
    mock_subprocess_run.assert_not_called()


def test_retry_command_success_first_attempt(mocker):
    mock_subprocess_run = mocker.patch("simtools.job_execution.job_manager.subprocess.run")
    mock_subprocess_run.return_value = None

    result = jm.retry_command("echo test", max_attempts=3, delay=1)

    assert result is True
    mock_subprocess_run.assert_called_once_with("echo test", shell=True, check=True, text=True)


def test_retry_command_success_second_attempt(mocker):
    mock_subprocess_run = mocker.patch("simtools.job_execution.job_manager.subprocess.run")
    mock_subprocess_run.side_effect = [subprocess.CalledProcessError(1, "echo test"), None]
    mock_sleep = mocker.patch("simtools.job_execution.job_manager.time.sleep")

    result = jm.retry_command("echo test", max_attempts=3, delay=5)

    assert result is True
    assert mock_subprocess_run.call_count == 2
    mock_sleep.assert_called_once_with(5)


def test_retry_command_failure_all_attempts(mocker):
    mock_subprocess_run = mocker.patch("simtools.job_execution.job_manager.subprocess.run")
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(1, "failing command")
    mock_sleep = mocker.patch("simtools.job_execution.job_manager.time.sleep")

    with pytest.raises(subprocess.CalledProcessError):
        jm.retry_command("failing command", max_attempts=2, delay=1)

    assert mock_subprocess_run.call_count == 2
    mock_sleep.assert_called_once_with(1)


def test_retry_command_default_parameters(mocker):
    mock_subprocess_run = mocker.patch("simtools.job_execution.job_manager.subprocess.run")
    mock_subprocess_run.return_value = None

    result = jm.retry_command("echo test")

    assert result is True
    mock_subprocess_run.assert_called_once_with("echo test", shell=True, check=True, text=True)
