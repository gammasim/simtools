"""Interface to workload managers to run jobs on a compute node."""

import logging
import stat
import subprocess
import time
from pathlib import Path

import simtools.utils.general as gen

logger = logging.getLogger(__name__)


class JobExecutionError(Exception):
    """Job execution error."""


def retry_command(command, max_attempts=3, delay=10):
    """
    Execute a shell command with retry logic for network-related failures.

    Parameters
    ----------
    command : str
        Shell command to execute.
    max_attempts : int
        Maximum number of retry attempts (default: 3).
    delay : int
        Delay in seconds between attempts (default: 10).

    Returns
    -------
    bool
        True if command succeeded, False if all attempts failed.

    Raises
    ------
    subprocess.CalledProcessError
        If command fails after all retry attempts.
    """
    for attempt in range(1, max_attempts + 1):
        logger.info(f"Attempt {attempt} of {max_attempts}: {command}")
        try:
            subprocess.run(command, shell=True, check=True, text=True)
            logger.info(f"Command succeeded on attempt {attempt}")
            return True
        except subprocess.CalledProcessError as exc:
            logger.warning(f"Command failed on attempt {attempt}")
            if attempt < max_attempts:
                logger.info(f"Waiting {delay}s before retry...")
                time.sleep(delay)
            else:
                logger.error(f"Command failed after {max_attempts} attempts")
                raise exc from None

    return False


def submit(
    command,
    out_file,
    err_file,
    configuration=None,
    application_log=None,
    runtime_environment=None,
    test=False,
):
    """
    Submit a job described by a command or a shell script.

    Allow to specify a runtime environment (e.g., Docker).

    Parameters
    ----------
    command: str
        Command or shell script to execute.
    out_file: str or Path
        Output stream (stdout if out_file and err_file are None).
    err_file: str or Path
        Error stream (stderr if out_file and err_file are None).
    configuration: dict
        Configuration for the 'command' execution.
    runtime_environment: list
        Command to run the application in the specified runtime environment.
    application_log: str or Path
        The log file of the actual application.
        Provided in order to print the log excerpt in case of run time error.
    test: bool
        Testing mode without sub submission.
    """
    command = _build_command(command, configuration, runtime_environment)

    logger.info(f"Submitting command {command}")
    logger.info(f"Job output/error streams {out_file} / {err_file}")

    if test:
        logger.info("Testing mode enabled")
        return None

    # disable pylint warning about not closing files here (explicitly closed in finally block)
    stdout = open(out_file, "w", encoding="utf-8") if out_file else subprocess.PIPE  # pylint: disable=consider-using-with
    stderr = open(err_file, "w", encoding="utf-8") if err_file else subprocess.PIPE  # pylint: disable=consider-using-with

    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            stdout=stdout,
            stderr=stderr,
        )

    except subprocess.CalledProcessError as exc:
        _raise_job_execution_error(exc, out_file, err_file, application_log)
    finally:
        if stdout != subprocess.PIPE:
            stdout.close()
        if stderr != subprocess.PIPE:
            stderr.close()

    return result


def _build_command(command, configuration=None, runtime_environment=None):
    """Build command to run in the specified runtime environment."""
    if isinstance(command, (str, Path)) and Path(command).is_file():
        command = Path(command)
        command.chmod(command.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
        command = str(command)

    if runtime_environment:
        if isinstance(runtime_environment, list):
            command = [*runtime_environment, command]
        else:
            command = [runtime_environment, command]

    if configuration:
        if isinstance(command, list):
            command = command + _convert_dict_to_args(configuration)
        else:
            command = [command, *_convert_dict_to_args(configuration)]

    return command


def _convert_dict_to_args(parameters):
    """
    Convert a dictionary of parameters to a list of command line arguments.

    Parameters
    ----------
    parameters : dict
        Dictionary containing parameters to convert.

    Returns
    -------
    list
        List of command line arguments.
    """
    args = []
    for key, value in parameters.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        elif isinstance(value, list):
            args.extend([f"--{key}", *(str(item) for item in value)])
        else:
            args.extend([f"--{key}", str(value)])
    return args


def _raise_job_execution_error(exc, out_file, err_file, application_log):
    """
    Raise job execution error with log excerpt.

    Parameters
    ----------
    exc: subprocess.CalledProcessError
        The caught exception.
    out_file: str or Path
        Output stream file path.
    err_file: str or Path
        Error stream file path.
    application_log: str or Path
        The log file of the actual application.
    """
    logger.error(f"Job execution failed with return code {exc.returncode}")

    if out_file:
        logger.error(f"Output log excerpt from {out_file}:\n{gen.get_log_excerpt(out_file)}")

    if err_file:
        logger.error(f"Error log excerpt from {err_file}:\n{gen.get_log_excerpt(err_file)}")

    if application_log:
        log = Path(application_log)
        if log.exists() and gen.get_file_age(log) < 5:
            logger.error(
                f"Application log excerpt from {application_log}:\n"
                f"{gen.get_log_excerpt(application_log)}"
            )

    raise JobExecutionError("See excerpt from log file above") from exc
