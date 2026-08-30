"""HTCondor execution backend using the Python bindings."""

import logging
import os
import pickle
import shlex
import shutil
import sys
import time
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from simtools.job_execution.backends.base import (
    BackendConfigurationError,
    BackendExecutionError,
    BackendSubmissionError,
)
from simtools.job_execution.job import JobResult, SubmissionHandle
from simtools.job_execution.worker import validate_job_id, write_job_payload

logger = logging.getLogger(__name__)

_RESERVED_SUBMIT_ATTRIBUTES = {"queue"}
_CONFIG_KEYS = {
    "request_cpus",
    "request_memory",
    "request_disk",
    "priority",
    "container_image",
    "container_target_dir",
    "python_executable",
    "environment_file",
    "poll_interval",
    "timeout",
    "cancel_on_interrupt",
    "keep_successful_artifacts",
    "extra_submit_attributes",
    "log_path",
}
_REQUEST_CPUS_ERROR = "request_cpus must be a positive integer."
_PRIORITY_ERROR = "priority must be an integer."
_DEFAULT_CONTAINER_PYTHON = "python"
_DEFAULT_CONTAINER_TARGET_DIR = "/simtools-run"
_CONTAINER_LAUNCHER = "/usr/bin/env"


class HTCondorBackend:
    """Submit one HTCondor process for every job specification."""

    supports_submit_only = True

    def __init__(self):
        self._htcondor = None
        self._schedd = None

    def _load_htcondor(self):
        if self._htcondor is not None:
            return self._htcondor
        try:
            import htcondor2 as htcondor  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise BackendConfigurationError(
                "The HTCondor backend requires gammasimtools[htcondor]."
            ) from exc
        self._htcondor = htcondor
        try:
            self._schedd = htcondor.Schedd()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            raise BackendConfigurationError(
                f"Cannot connect to the HTCondor schedd: {exc}"
            ) from exc
        return htcondor

    @staticmethod
    def _validate_config(config):
        unknown = set(config) - _CONFIG_KEYS
        if unknown:
            names = ", ".join(sorted(unknown))
            raise BackendConfigurationError(f"Unknown HTCondor configuration key(s): {names}.")
        HTCondorBackend._validate_request_cpus(config)
        HTCondorBackend._validate_priority(config)
        HTCondorBackend._validate_resource_sizes(config)
        HTCondorBackend._validate_python_executable(config)
        HTCondorBackend._validate_container_target_dir(config)
        HTCondorBackend._validate_extra_attributes(config)
        HTCondorBackend._validate_timing(config)
        HTCondorBackend._validate_paths(config)

    @staticmethod
    def _validate_request_cpus(config):
        """Validate the default CPU request."""
        raw_request_cpus = config.get("request_cpus", 1)
        try:
            request_cpus = int(raw_request_cpus)
        except (TypeError, ValueError) as exc:
            raise BackendConfigurationError(_REQUEST_CPUS_ERROR) from exc
        if isinstance(raw_request_cpus, float) and not raw_request_cpus.is_integer():
            raise BackendConfigurationError(_REQUEST_CPUS_ERROR)
        if isinstance(raw_request_cpus, bool):
            raise BackendConfigurationError(_REQUEST_CPUS_ERROR)
        if request_cpus < 1:
            raise BackendConfigurationError(_REQUEST_CPUS_ERROR)

    @staticmethod
    def _validate_priority(config):
        """Validate the scheduler job priority."""
        raw_priority = config.get("priority", 0)
        try:
            priority = int(raw_priority)
        except (TypeError, ValueError) as exc:
            raise BackendConfigurationError(_PRIORITY_ERROR) from exc
        if isinstance(raw_priority, bool) or (
            isinstance(raw_priority, float) and not raw_priority.is_integer()
        ):
            raise BackendConfigurationError(_PRIORITY_ERROR)
        if str(raw_priority).strip() != str(priority):
            raise BackendConfigurationError(_PRIORITY_ERROR)

    @staticmethod
    def _validate_resource_sizes(config):
        """Validate optional HTCondor memory and disk expressions."""
        for key in ("request_memory", "request_disk"):
            value = config.get(key)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise BackendConfigurationError(f"{key} must be a non-empty string.")

    @staticmethod
    def _validate_python_executable(config):
        """Validate the optional Python command used inside container images."""
        python_executable = config.get("python_executable", _DEFAULT_CONTAINER_PYTHON)
        if not isinstance(python_executable, str) or not python_executable.strip():
            raise BackendConfigurationError("python_executable must be a non-empty string.")

    @staticmethod
    def _validate_container_target_dir(config):
        """Validate the directory used for the container scratch mount."""
        target_dir = config.get("container_target_dir", _DEFAULT_CONTAINER_TARGET_DIR)
        if not isinstance(target_dir, str) or not target_dir.strip():
            raise BackendConfigurationError("container_target_dir must be a non-empty path.")
        if not Path(target_dir).is_absolute():
            raise BackendConfigurationError("container_target_dir must be an absolute path.")

    @staticmethod
    def _validate_extra_attributes(config):
        """Validate custom scheduler attributes."""
        extra_attributes = config.get("extra_submit_attributes", {})
        if not isinstance(extra_attributes, dict):
            raise BackendConfigurationError("extra_submit_attributes must be a mapping.")
        return extra_attributes

    @staticmethod
    def _add_extra_submit_attributes(submit_values, extra_attributes):
        """Add custom attributes without allowing fixed values to be replaced."""
        protected = _RESERVED_SUBMIT_ATTRIBUTES | {name.lower() for name in submit_values}
        protected &= {str(name).lower() for name in extra_attributes}
        if protected:
            names = ", ".join(sorted(protected))
            raise BackendConfigurationError(f"Protected submit attribute(s): {names}.")
        submit_values.update(extra_attributes)

    @staticmethod
    def _validate_timing(config):
        """Validate polling, timeout, and boolean execution settings."""
        try:
            poll_interval = float(config.get("poll_interval", 60))
        except (TypeError, ValueError) as exc:
            raise BackendConfigurationError("poll_interval must be a positive number.") from exc
        if poll_interval <= 0:
            raise BackendConfigurationError("poll_interval must be a positive number.")
        timeout = config.get("timeout")
        if timeout is not None:
            try:
                timeout = float(timeout)
            except (TypeError, ValueError) as exc:
                raise BackendConfigurationError(
                    "timeout must be a positive number or null."
                ) from exc
            if timeout <= 0:
                raise BackendConfigurationError("timeout must be a positive number or null.")
        for key in ("cancel_on_interrupt", "keep_successful_artifacts"):
            if key in config and not isinstance(config[key], bool):
                raise BackendConfigurationError(f"{key} must be boolean.")

    @staticmethod
    def _validate_paths(config):
        """Validate configured shared files."""
        for key in ("container_image", "environment_file"):
            value = config.get(key)
            if value is not None and not Path(value).is_file():
                raise BackendConfigurationError(f"Configured {key} does not exist: {value}")

    @staticmethod
    def _read_environment_file(path, bind_paths=(), python_path=None):
        """Convert a simple dotenv file into an HTCondor environment value."""
        if path is None and not bind_paths and python_path is None:
            return None
        entries = HTCondorBackend._read_environment_entries(path) if path is not None else {}
        HTCondorBackend._add_corsika_interaction_table_bind(entries)
        if python_path is not None:
            HTCondorBackend._add_python_path(entries, python_path)
        for bind_path in bind_paths:
            HTCondorBackend._add_apptainer_bind_path(entries, bind_path)
        return ";".join(f"{key}={value}" for key, value in entries.items())

    @staticmethod
    def _source_checkout_path():
        """Return the current source checkout path when simtools runs from one."""
        source_path = Path(__file__).resolve().parents[3]
        project_file = source_path.parent / "pyproject.toml"
        if project_file.is_file() and (source_path / "simtools").is_dir():
            return source_path
        return None

    @staticmethod
    def _add_python_path(entries, source_path):
        """Prepend the source checkout to the worker Python module path."""
        paths = [str(source_path)]
        if entries.get("PYTHONPATH"):
            paths.extend(path for path in entries["PYTHONPATH"].split(os.pathsep) if path)
        entries["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))

    @staticmethod
    def _read_environment_entries(path):
        """Read and normalize entries from a dotenv file."""
        entries = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            entry = HTCondorBackend._parse_environment_line(line, path)
            if entry is not None:
                entries[entry[0]] = entry[1]
        return entries

    @staticmethod
    def _parse_environment_line(line, path):
        """Parse one dotenv line, returning ``None`` for comments and blanks."""
        line = line.strip()
        if not line or line.startswith("#"):
            return None
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise BackendConfigurationError(
                f"Invalid environment entry in {path}: expected KEY=VALUE."
            )
        key, value = line.split("=", 1)
        key = key.strip()
        value = HTCondorBackend._strip_environment_comment(value).strip().strip("\"'")
        if not key:
            raise BackendConfigurationError(f"Invalid empty environment key in {path}.")
        return key, value

    @staticmethod
    def _add_corsika_interaction_table_bind(entries):
        """Make CORSIKA interaction tables visible inside Apptainer jobs."""
        table_path = entries.get("SIMTOOLS_CORSIKA_INTERACTION_TABLE_PATH")
        if not table_path:
            return
        HTCondorBackend._add_apptainer_bind_path(entries, table_path)

    @staticmethod
    def _add_apptainer_bind_path(entries, bind_path):
        """Add an Apptainer bind without duplicating nested container destinations."""
        if not bind_path:
            return
        bind_paths = [path for path in entries.get("APPTAINER_BINDPATH", "").split(",") if path]
        _, candidate_destination = HTCondorBackend._apptainer_bind_paths(bind_path)
        existing_destinations = [
            HTCondorBackend._apptainer_bind_paths(existing)[1] for existing in bind_paths
        ]
        if any(
            candidate_destination == destination
            or candidate_destination.is_relative_to(destination)
            for destination in existing_destinations
        ):
            return
        bind_paths = [
            existing
            for existing, destination in zip(bind_paths, existing_destinations, strict=True)
            if not destination.is_relative_to(candidate_destination)
        ]
        bind_paths.append(str(bind_path))
        entries["APPTAINER_BINDPATH"] = ",".join(bind_paths)

    @staticmethod
    def _apptainer_bind_paths(bind_path):
        """Return normalized source and destination paths for an Apptainer bind entry."""
        source, separator, destination_and_options = str(bind_path).partition(":")
        destination = destination_and_options.split(":", 1)[0] if separator else source
        return (
            Path(source).expanduser().resolve(),
            Path(destination or source).expanduser().resolve(),
        )

    @staticmethod
    def _strip_environment_comment(value):
        """Remove an unquoted inline comment from an environment value."""
        quote = None
        escaped = False
        for index, character in enumerate(value):
            if escaped:
                escaped = False
            elif character == "\\" and quote == '"':
                escaped = True
            else:
                quote = HTCondorBackend._update_environment_quote(quote, character)
            if HTCondorBackend._is_environment_comment(value, index, character, quote, escaped):
                return value[:index]
        return value

    @staticmethod
    def _update_environment_quote(quote, character):
        """Update the active quote character for an environment value."""
        if character not in "\"'":
            return quote
        if quote == character:
            return None
        return character if quote is None else quote

    @staticmethod
    def _is_environment_comment(value, index, character, quote, escaped):
        """Return whether a character starts an unquoted environment comment."""
        return (
            character == "#"
            and quote is None
            and not escaped
            and (index == 0 or value[index - 1].isspace())
        )

    def submit(self, job_specs, options):
        """Create and submit a shared-filesystem HTCondor cluster."""
        config = self._build_config(options)
        self._validate_config(config)
        jobs = self._prepare_jobs(job_specs, options, config)
        working_directory = Path.cwd().resolve()
        if not jobs:
            return SubmissionHandle(
                backend="htcondor",
                work_dir=Path(options.work_dir or Path.cwd()),
                job_ids=(),
            )
        htcondor = self._load_htcondor()
        self._validate_job_resources(jobs, config)
        work_dir = self._create_work_dir(options)
        self._serialize_jobs(jobs, work_dir)
        event_log = self._resolve_event_log(config, work_dir)
        submit_values, resource_defaults, resource_keys = self._build_submit_values(
            config, jobs, work_dir, event_log, working_directory
        )
        itemdata = self._build_itemdata(jobs, resource_defaults, resource_keys)
        try:
            submission = htcondor.Submit(submit_values)
            result = self._schedd.submit(submission, itemdata=iter(itemdata))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            raise BackendSubmissionError(f"HTCondor submission failed: {exc}") from exc
        handle = self._make_handle(result, jobs, work_dir, event_log, config, working_directory)
        logger.info("Submitted %d HTCondor jobs as cluster %d", len(jobs), handle.scheduler_id)
        return handle

    @staticmethod
    def _prepare_jobs(job_specs, options, config):
        """Materialize jobs and normalize worker-specific settings."""
        jobs = list(job_specs)
        prepared = []
        for job in jobs:
            resources = dict(job.resources)
            if resources.get("container_image"):
                resources["container_image"] = str(
                    Path(resources["container_image"]).expanduser().resolve()
                )
            initializer = job.initializer if job.initializer is not None else options.initializer
            initargs = job.initargs if job.initializer is not None else tuple(options.initargs)
            command = job.command
            uses_container = bool(config.get("container_image") or resources.get("container_image"))
            if uses_container and command and command[0] == sys.executable:
                command = (
                    config.get("python_executable", _DEFAULT_CONTAINER_PYTHON),
                    *command[1:],
                )
            prepared.append(
                replace(
                    job,
                    command=command,
                    resources=resources,
                    initializer=initializer,
                    initargs=initargs,
                )
            )
        return prepared

    @staticmethod
    def _build_config(options):
        """Normalize filesystem paths in HTCondor backend configuration."""
        config = dict(options.backend_config)
        for key in ("container_image", "environment_file"):
            if config.get(key):
                config[key] = str(Path(config[key]).expanduser().resolve())
        return config

    @staticmethod
    def _validate_job_resources(jobs, config=None):
        """Validate supported per-job resource overrides."""
        config = config or {}
        for job in jobs:
            unknown = set(job.resources) - {
                "request_cpus",
                "request_memory",
                "request_disk",
                "priority",
                "container_image",
            }
            if unknown:
                raise BackendConfigurationError(
                    f"Unknown resource key(s) for job {job.job_id}: " + ", ".join(sorted(unknown))
                )
            HTCondorBackend._validate_request_cpus(job.resources)
            HTCondorBackend._validate_priority(job.resources)
            HTCondorBackend._validate_resource_sizes(job.resources)
            image = job.resources.get("container_image")
            if image is not None and not Path(image).is_file():
                raise BackendConfigurationError(
                    f"Configured container_image does not exist: {image}"
                )
        has_image_override = any("container_image" in job.resources for job in jobs)
        if (
            has_image_override
            and not config.get("container_image")
            and any(not job.resources.get("container_image") for job in jobs)
        ):
            raise BackendConfigurationError(
                "Every job must define container_image when per-job container overrides are used."
            )

    @staticmethod
    def _create_work_dir(options):
        """Create the shared, private run directory."""
        root = Path(options.work_dir or Path.cwd() / "simtools-jobs").expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        work_dir = root / f"htcondor-{uuid.uuid4().hex}"
        work_dir.mkdir(mode=0o700)
        for directory in (
            work_dir / "inputs",
            work_dir / "results",
            work_dir / "stdout",
            work_dir / "stderr",
            work_dir / "logs",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return work_dir

    @staticmethod
    def _serialize_jobs(jobs, work_dir):
        """Serialize one payload per job."""
        for job in jobs:
            try:
                validate_job_id(job.job_id)
                write_job_payload(job, work_dir / "inputs" / f"{job.job_id}.pkl")
            except (OSError, pickle.PickleError, TypeError, AttributeError, ValueError) as exc:
                raise BackendSubmissionError(
                    f"Cannot serialize HTCondor job {job.job_id}: {exc}"
                ) from exc

    @staticmethod
    def _resolve_event_log(config, work_dir):
        """Resolve and create the scheduler event-log path."""
        event_log = Path(config.get("log_path") or work_dir / "scheduler.log")
        if not event_log.is_absolute():
            event_log = work_dir / event_log
        event_log.parent.mkdir(parents=True, exist_ok=True)
        return event_log

    def _build_submit_values(self, config, jobs, work_dir, event_log, working_directory=None):
        """Build the submit description and per-job resource metadata."""
        working_directory = Path(working_directory or Path.cwd()).resolve()
        resource_defaults, resource_keys = self._get_resource_data(config, jobs)
        uses_container = bool(config.get("container_image") or "container_image" in resource_keys)
        submit_values = self._build_worker_submit_values(
            config, work_dir, event_log, working_directory, uses_container
        )
        self._add_resource_submit_values(submit_values, config, resource_keys)
        self._add_container_submit_values(submit_values, config, uses_container)
        source_path = self._source_checkout_path()
        bind_paths = self._build_container_bind_paths(
            uses_container, source_path, work_dir, working_directory, jobs
        )
        environment = self._read_environment_file(
            config.get("environment_file"),
            bind_paths=bind_paths,
            python_path=source_path,
        )
        if environment:
            submit_values["environment"] = environment
        self._add_extra_submit_attributes(submit_values, self._validate_extra_attributes(config))
        return submit_values, resource_defaults, resource_keys

    @staticmethod
    def _get_resource_data(config, jobs):
        """Return default and per-job resource settings."""
        resource_defaults = {
            "request_cpus": config.get("request_cpus", 1),
            "request_memory": config.get("request_memory") or "",
            "request_disk": config.get("request_disk") or "",
            "priority": config.get("priority") if config.get("priority") is not None else 0,
            "container_image": config.get("container_image") or "",
        }
        resource_keys = tuple(
            key for key in resource_defaults if any(key in job.resources for job in jobs)
        )
        return resource_defaults, resource_keys

    @staticmethod
    def _build_worker_submit_values(config, work_dir, event_log, working_directory, uses_container):
        """Build the fixed HTCondor submit attributes for the worker process."""
        python_executable = config.get("python_executable", _DEFAULT_CONTAINER_PYTHON)
        worker_arguments = [
            *([shlex.quote(python_executable)] if uses_container else []),
            shlex.quote("-m"),
            shlex.quote("simtools.job_execution.worker"),
            shlex.quote("--run-directory"),
            shlex.quote(str(work_dir)),
            shlex.quote("--job-id"),
            "$(job_id)",
            shlex.quote("--log-file"),
            str(work_dir / "logs" / "$(job_id).log"),
        ]
        return {
            "executable": _CONTAINER_LAUNCHER if uses_container else sys.executable,
            "arguments": " ".join(worker_arguments),
            "initialdir": str(working_directory),
            "output": str(work_dir / "stdout" / "$(job_id).out"),
            "error": str(work_dir / "stderr" / "$(job_id).err"),
            "log": str(event_log),
            "should_transfer_files": "NO",
            "request_cpus": str(config.get("request_cpus", 1)),
        }

    @staticmethod
    def _add_resource_submit_values(submit_values, config, resource_keys):
        """Add resource defaults and explicit values to a submit description."""
        for key in resource_keys:
            submit_values[key] = f"$({key})"
        for key in ("request_memory", "request_disk", "priority", "container_image"):
            if config.get(key) is not None:
                submit_values[key] = str(config[key])

    @staticmethod
    def _add_container_submit_values(submit_values, config, uses_container):
        """Add container-specific submit attributes when a container is used."""
        if uses_container:
            submit_values["universe"] = "container"
            submit_values["container_target_dir"] = config.get(
                "container_target_dir", _DEFAULT_CONTAINER_TARGET_DIR
            )

    @staticmethod
    def _build_container_bind_paths(uses_container, source_path, work_dir, working_directory, jobs):
        """Return minimized bind paths needed by container jobs."""
        if not uses_container:
            return ()
        bind_paths = [work_dir.parent, working_directory]
        if source_path is not None:
            bind_paths.append(source_path)
        bind_paths.extend(
            Path(mount_path).expanduser().resolve()
            for job in jobs
            for mount_path in job.mount_paths
        )
        bind_paths.extend(
            Path(output_path).expanduser().resolve().parent
            for job in jobs
            for output_path in job.output_paths
        )
        return HTCondorBackend._minimize_bind_paths(bind_paths)

    @staticmethod
    def _minimize_bind_paths(bind_paths):
        """Remove duplicate and nested paths from container bind paths."""
        minimized = []
        for bind_path in sorted(
            {Path(path) for path in bind_paths}, key=lambda path: len(path.parts)
        ):
            if not any(bind_path.is_relative_to(parent) for parent in minimized):
                minimized.append(bind_path)
        return minimized

    @staticmethod
    def _build_itemdata(jobs, resource_defaults, resource_keys):
        """Build item data rows consumed by the HTCondor bindings."""
        itemdata = []
        for job in jobs:
            row = {"job_id": job.job_id}
            for key in resource_keys:
                row[key] = str(job.resources.get(key, resource_defaults[key]))
            itemdata.append(row)
        return itemdata

    @staticmethod
    def _make_handle(result, jobs, work_dir, event_log, config, working_directory):
        """Create a submission handle from a scheduler result."""
        cluster_id = int(result.cluster())
        first_proc = getattr(result, "first_proc", 0)
        first_proc = first_proc() if callable(first_proc) else first_proc
        first_proc = int(first_proc)
        process_ids = {job.job_id: first_proc + index for index, job in enumerate(jobs)}
        return SubmissionHandle(
            backend="htcondor",
            work_dir=work_dir,
            job_ids=tuple(job.job_id for job in jobs),
            scheduler_id=cluster_id,
            process_ids=process_ids,
            metadata={
                "poll_interval": float(config.get("poll_interval", 60)),
                "timeout": config.get("timeout"),
                "cancel_on_interrupt": bool(config.get("cancel_on_interrupt", False)),
                "keep_successful_artifacts": bool(config.get("keep_successful_artifacts", False)),
                "request_cpus": config.get("request_cpus", 1),
                "request_memory": config.get("request_memory"),
                "request_disk": config.get("request_disk"),
                "priority": config.get("priority"),
                "container_image": config.get("container_image"),
                "environment_file": str(config["environment_file"])
                if config.get("environment_file")
                else None,
                "job_log_dir": str(work_dir / "logs"),
                "event_log": str(event_log),
                "working_directory": str(working_directory),
                "submitted_at": datetime.now(UTC).isoformat(),
                "indices": {job.job_id: job.index for job in jobs},
            },
        )

    def wait(self, submission):
        """Wait for all HTCondor processes and load their result files."""
        if not submission.job_ids:
            return []
        self._load_htcondor()
        failures = self._wait_for_processes(submission)
        results, result_failures = self._load_results(submission)
        failures.extend(result_failures)
        failures.extend(self._missing_output_failures(submission))
        if failures:
            raise BackendExecutionError("HTCondor job failure(s): " + "; ".join(failures))
        self._cleanup_successful_artifacts(submission)
        return sorted(results, key=lambda result: result.index)

    @staticmethod
    def _missing_output_failures(submission):
        """Report missing declared outputs before transient artifacts are removed."""
        expected = submission.metadata.get("expected_outputs", {})
        if isinstance(expected, list):
            expected = {"unknown": expected}
        return [
            f"job {job_id}: missing expected output {path}"
            for job_id, paths in expected.items()
            for path in paths
            if not Path(path).exists()
        ]

    @staticmethod
    def _cleanup_successful_artifacts(submission):
        """Remove transient payload and stream files after successful execution."""
        if submission.metadata.get("keep_successful_artifacts"):
            return
        for directory in ("inputs", "results", "stdout", "stderr"):
            shutil.rmtree(submission.work_dir / directory, ignore_errors=True)

    def _wait_for_processes(self, submission):
        """Wait for scheduler terminal events and return process failures."""
        event_log_path = Path(
            submission.metadata.get("event_log", submission.work_dir / "scheduler.log")
        )
        try:
            event_log = self._htcondor.JobEventLog(str(event_log_path))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            raise BackendExecutionError(f"Cannot open HTCondor event log: {exc}") from exc

        remaining = set(submission.process_ids.values())
        failures = []
        start_time = time.monotonic()
        timeout = submission.metadata.get("timeout")
        timeout = float(timeout) if timeout is not None else None
        terminal_types = {"JOB_TERMINATED", "JOB_ABORTED", "JOB_HELD", "CLUSTER_REMOVE"}
        while remaining:
            if timeout is not None and time.monotonic() - start_time >= timeout:
                failures.extend(f"process {proc}: timeout" for proc in sorted(remaining))
                self._cancel_after_wait_failure(submission, failures)
                break
            event = self._next_event(event_log, submission.metadata["poll_interval"])
            if event is None:
                continue
            event_failures = self._process_event(
                event, submission.scheduler_id, remaining, terminal_types
            )
            failures.extend(event_failures)
            if self._is_held_event(event, submission.scheduler_id, submission.process_ids.values()):
                self._cancel_after_wait_failure(submission, failures)
        return failures

    def _cancel_after_wait_failure(self, submission, failures):
        """Cancel a cluster that cannot complete normally while preserving diagnostics."""
        try:
            self.cancel(submission)
        except BackendExecutionError as exc:
            failures.append(f"Cancellation failed: {exc}")

    @staticmethod
    def _is_held_event(event, scheduler_id, process_ids):
        """Return whether an event reports a held process in this submission."""
        if int(getattr(event, "cluster", -1)) != scheduler_id:
            return False
        event_type = getattr(event, "type", "")
        return (
            int(getattr(event, "proc", -1)) in process_ids
            and getattr(event_type, "name", str(event_type)).upper() == "JOB_HELD"
        )

    @staticmethod
    def _next_event(event_log, poll_interval):
        """Read one event using the integer deadline required by htcondor2."""
        try:
            deadline = max(1, int(poll_interval))
            return next(event_log.events(stop_after=deadline))
        except StopIteration:
            return None

    @staticmethod
    def _process_event(event, scheduler_id, remaining, terminal_types):
        """Update remaining processes from one scheduler event."""
        if int(getattr(event, "cluster", -1)) != scheduler_id:
            return []
        event_type = getattr(event, "type", "")
        event_name = getattr(event_type, "name", str(event_type)).upper()
        if event_name == "CLUSTER_REMOVE":
            failures = [f"process {proc}: CLUSTER_REMOVE" for proc in sorted(remaining)]
            remaining.clear()
            return failures
        proc = int(getattr(event, "proc", -1))
        if proc not in remaining or event_name not in terminal_types:
            return []
        remaining.remove(proc)
        if event_name == "JOB_TERMINATED" and int(event.get("ReturnValue", 1)) == 0:
            return []
        return [f"process {proc}: {event_name} ({dict(event)})"]

    @staticmethod
    def _load_results(submission):
        """Load worker result payloads and return values and failures."""
        results = []
        failures = []
        for job_id, process_id in submission.process_ids.items():
            result_path = submission.work_dir / "results" / f"{job_id}.pkl"
            if not result_path.exists():
                stderr_path = submission.work_dir / "stderr" / f"{job_id}.err"
                failures.append(
                    f"process {process_id}: missing result {result_path}; stderr {stderr_path}"
                )
                continue
            try:
                with result_path.open("rb") as handle:
                    payload = pickle.load(handle)
            except (OSError, pickle.PickleError, EOFError, ValueError) as exc:
                failures.append(f"process {process_id}: unreadable result {result_path}: {exc}")
                continue
            if not isinstance(payload, dict):
                failures.append(f"process {process_id}: invalid result payload {result_path}")
                continue
            if not payload.get("ok"):
                failures.append(f"process {process_id}: {payload.get('message')}")
                continue
            index = submission.metadata.get("indices", {}).get(job_id)
            if index is None:
                failures.append(f"process {process_id}: missing job index for {job_id}")
                continue
            results.append(JobResult(job_id, index, payload.get("value")))
        return results, failures

    def cancel(self, submission):
        """Remove all processes in an active cluster."""
        if submission.scheduler_id is None:
            return
        self._load_htcondor()
        try:
            self._schedd.act(
                self._htcondor.JobAction.Remove, f"ClusterId == {submission.scheduler_id}"
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            raise BackendExecutionError(f"Cannot cancel HTCondor cluster: {exc}") from exc
