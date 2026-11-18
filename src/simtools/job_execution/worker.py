"""Execute one serialized job specification for a remote backend."""

import argparse
import copy
import inspect
import json
import logging
import os
import pickle
import re
import shlex
import stat
import subprocess
import tempfile
import time
import traceback
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path

from simtools import constants
from simtools.job_execution.job import JobSpec
from simtools.model_repository.reader import create_model_reader_from_source_config
from simtools.runtime.context import RunContext
from simtools.runtime.dependencies import stage_corsika_interaction_tables

logger = logging.getLogger(__name__)

_PAYLOAD_VERSION = 1
_JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--scratch-directory", type=Path)
    return parser


def _configure_worker_logging(log_file):
    """Configure INFO-level logging for one remote worker."""
    if log_file is None:
        return None
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers:
        handler.setLevel(logging.WARNING)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root_logger.addHandler(file_handler)
    return file_handler


def execute_job_spec(job_spec):
    """Execute one Python or command job specification."""
    context = _initialize_runtime(job_spec)
    if job_spec.initializer is not None:
        job_spec.initializer(*job_spec.initargs)
    if job_spec.function is not None:
        if context is not None and "context" in inspect.signature(job_spec.function).parameters:
            return job_spec.function(job_spec.item, context=context)
        return job_spec.function(job_spec.item)
    return execute_command(job_spec)[0]


def execute_command(job_spec):
    """Run a command job with its configured streams and environment.

    Returns
    -------
    tuple
        Return code, wall-clock runtime in seconds, and the captured standard output and
        standard error (None unless captured).
    """
    command = job_spec.command
    if isinstance(command, (str, Path)) and Path(command).is_file():
        command_path = Path(command)
        command_path.chmod(command_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
        command = str(command_path)
    if job_spec.options.runtime_environment:
        runtime = job_spec.options.runtime_environment
        prefix = list(runtime) if isinstance(runtime, (tuple, list)) else [runtime]
        command = [*prefix, command]
    if job_spec.options.configuration:
        args = _configuration_arguments(job_spec.options.configuration)
        if isinstance(command, str):
            command = f"{command} {shlex.join(args)}"
        else:
            command = [*command, *args]
    environment = os.environ.copy()
    environment.update(job_spec.options.env or {})
    started = time.perf_counter()
    with ExitStack() as streams:
        stdin = job_spec.options.stdin
        if isinstance(stdin, (str, Path)):
            stdin = streams.enter_context(Path(stdin).open(encoding="utf-8"))
        stdout = _stream(streams, job_spec.options.stdout, job_spec.options.capture_output)
        stderr = _stream(streams, job_spec.options.stderr, job_spec.options.capture_output)
        result = subprocess.run(
            command,
            shell=isinstance(command, str),
            check=job_spec.options.check,
            text=True,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            env=environment,
        )
    return (
        result.returncode,
        time.perf_counter() - started,
        result.stdout if isinstance(result.stdout, str) else None,
        result.stderr if isinstance(result.stderr, str) else None,
    )


def _stream(streams, path, capture_output):
    """Return the subprocess stream for a log file (closed with ``streams``) or a pipe."""
    if path is not None:
        return streams.enter_context(Path(path).open("w", encoding="utf-8"))
    return subprocess.PIPE if capture_output else None


def _configuration_arguments(configuration):
    """Convert application configuration to command-line arguments."""
    arguments = []
    for key, value in configuration.items():
        if isinstance(value, bool):
            if value:
                arguments.append(f"--{key}")
        elif isinstance(value, list):
            arguments.extend(_list_arguments(key, value))
        else:
            arguments.extend((f"--{key}", str(value)))
    return arguments


def _list_arguments(key, value):
    """Convert a list configuration value (JSON objects, repeated groups or values)."""
    if not value:
        return []
    if isinstance(value[0], dict):
        return [f"--{key}", json.dumps(value)]
    if isinstance(value[0], list):
        return [token for group in value for token in (f"--{key}", *(str(item) for item in group))]
    return [f"--{key}", *(str(item) for item in value)]


def _initialize_runtime(job_spec):
    """Restore the submitting application's runtime for function jobs."""
    if job_spec.context_spec is None:
        return None
    return RunContext.from_spec(
        job_spec.context_spec, model_reader=create_model_reader_from_source_config
    )


@contextmanager
def _stage_tables_for_job(job_spec, scratch_directory=None):
    """Stage and verify CORSIKA interaction tables for one CORSIKA simulation job."""
    software, he_model, le_model = _corsika_job_settings(job_spec)
    source_path = _corsika_table_source(job_spec)
    if not software or not software.startswith("corsika") or not source_path:
        yield None
        return

    source_path = Path(source_path)
    if (source_path / "interaction-tables").is_dir():
        source_path /= "interaction-tables"
    scratch_path = scratch_directory or os.environ.get("_CONDOR_SCRATCH_DIR") or None
    with tempfile.TemporaryDirectory(prefix="simtools-corsika-tables-", dir=scratch_path) as temp:
        destination = Path(temp) / "interaction-tables"
        stage_corsika_interaction_tables(source_path, destination, he_model, le_model)
        logger.info(
            "Staged verified CORSIKA tables for %s/%s from %s to %s",
            he_model,
            le_model,
            source_path,
            destination,
        )
        yield destination, _job_with_table_path(job_spec, destination)


def _corsika_table_source(job_spec):
    """Return the configured shared interaction-table source for a job."""
    configuration = job_spec.options.configuration or {}
    source = configuration.get("corsika_interaction_table_path")
    source = source or _command_option(
        _command_tokens(job_spec.command), "corsika_interaction_table_path"
    )
    context_args = (job_spec.context_spec or {}).get("args", {})
    environment = job_spec.options.env or {}
    return (
        source
        or context_args.get("corsika_interaction_table_path")
        or environment.get("SIMTOOLS_CORSIKA_INTERACTION_TABLE_PATH")
        or os.environ.get("SIMTOOLS_CORSIKA_INTERACTION_TABLE_PATH")
    )


def _corsika_job_settings(job_spec):
    """Return simulation software and hadronic models selected by a job."""
    names = ("simulation_software", "corsika_he_interaction", "corsika_le_interaction")
    settings = (job_spec.context_spec or {}).get("args", {}).copy()
    tokens = _command_tokens(job_spec.command)
    for name in names:
        value = _command_option(tokens, name)
        if value is not None:
            settings[name] = value
    settings.update(job_spec.options.configuration or {})
    defaults = {
        "simulation_software": os.environ.get("SIMTOOLS_SIMULATION_SOFTWARE", ""),
        "corsika_he_interaction": os.environ.get(
            "SIMTOOLS_CORSIKA_HE_INTERACTION", constants.CORSIKA_HE_INTERACTION
        ),
        "corsika_le_interaction": os.environ.get(
            "SIMTOOLS_CORSIKA_LE_INTERACTION", constants.CORSIKA_LE_INTERACTION
        ),
    }
    return tuple(str(settings.get(name) or defaults[name]).lower() for name in names)


def _command_tokens(command):
    """Return a job command as tokens for option lookup."""
    if isinstance(command, str):
        return shlex.split(command)
    if isinstance(command, Path):
        return [str(command)]
    return list(command or ())


def _command_option(tokens, name):
    """Return the last value for one command-line option, if present."""
    option = f"--{name}"
    value = None
    for index, token in enumerate(tokens):
        if token == option and index + 1 < len(tokens):
            value = tokens[index + 1]
        elif token.startswith(f"{option}="):
            value = token.partition("=")[2]
    return value


def _job_with_table_path(job_spec, table_path):
    """Override shared table paths in the worker environment and job arguments."""
    environment = dict(job_spec.options.env or {})
    environment["SIMTOOLS_CORSIKA_INTERACTION_TABLE_PATH"] = str(table_path)
    configuration = dict(job_spec.options.configuration or {})
    configuration["corsika_interaction_table_path"] = str(table_path)
    options = replace(job_spec.options, env=environment, configuration=configuration)
    context_spec = copy.deepcopy(job_spec.context_spec)
    if context_spec is not None:
        context_spec.setdefault("args", {})["corsika_interaction_table_path"] = str(table_path)
    return replace(job_spec, options=options, context_spec=context_spec)


def write_job_payload(job_spec, path):
    """Serialize a versioned worker payload."""
    payload = {
        "payload_version": _PAYLOAD_VERSION,
        "job_spec": job_spec,
    }
    with Path(path).open("wb") as handle:
        pickle.dump(payload, handle)


def _load_job_payload(path):
    """Load and validate one trusted worker payload."""
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or payload.get("payload_version") != _PAYLOAD_VERSION:
        raise ValueError("Unsupported worker payload format.")
    job_spec = payload.get("job_spec")
    if not isinstance(job_spec, JobSpec):
        raise TypeError("Worker payload does not contain a JobSpec.")
    return job_spec


def load_job_payload(path):
    """Load a versioned JobSpec payload from a trusted execution work directory."""
    return _load_job_payload(path)


def _job_file(run_directory, directory, job_id):
    """Resolve a job-owned path below the private run directory."""
    validate_job_id(job_id)
    return Path(run_directory) / directory / f"{job_id}.pkl"


def validate_job_id(job_id):
    """Validate a job ID used in worker-owned file names."""
    if not isinstance(job_id, str) or not _JOB_ID_PATTERN.fullmatch(job_id):
        raise ValueError(f"Invalid job ID: {job_id!r}.")


def _write_result(path, payload):
    """Write a worker result atomically."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle)
    temporary.replace(path)


def run(run_directory, job_id, log_file=None, scratch_directory=None):
    """Run one serialized job and write its result or failure record."""
    run_directory = Path(run_directory)
    input_path = _job_file(run_directory, "inputs", job_id)
    result_path = _job_file(run_directory, "results", job_id)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = None
    previous_table_path = os.environ.get("SIMTOOLS_CORSIKA_INTERACTION_TABLE_PATH")
    try:
        file_handler = _configure_worker_logging(log_file)
        job_spec = _load_job_payload(input_path)
        with _stage_tables_for_job(job_spec, scratch_directory) as staged_tables:
            if staged_tables is not None:
                table_path, job_spec = staged_tables
                os.environ["SIMTOOLS_CORSIKA_INTERACTION_TABLE_PATH"] = str(table_path)
            value = execute_job_spec(job_spec)
        _write_result(result_path, {"ok": True, "value": value})
        return 0
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception("Job %s failed", job_id)
        payload = {
            "ok": False,
            "exception": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_result(result_path, payload)
        return 1
    finally:
        if previous_table_path is None:
            os.environ.pop("SIMTOOLS_CORSIKA_INTERACTION_TABLE_PATH", None)
        else:
            os.environ["SIMTOOLS_CORSIKA_INTERACTION_TABLE_PATH"] = previous_table_path
        if file_handler is not None:
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()


def main():
    """Run the worker CLI."""
    args = _parser().parse_args()
    raise SystemExit(run(args.run_directory, args.job_id, args.log_file, args.scratch_directory))


if __name__ == "__main__":
    main()
