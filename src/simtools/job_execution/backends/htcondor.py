"""HTCondor execution backend using the Python bindings."""

import logging
import pickle
import platform
import shlex
import shutil
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path

from simtools.job_execution.backends.base import (
    BackendConfigurationError,
    BackendExecutionError,
    BackendSubmissionError,
)
from simtools.job_execution.job import JobResult, SubmissionHandle
from simtools.version import __version__ as simtools_version

logger = logging.getLogger(__name__)

_PROTECTED_ATTRIBUTES = {
    "executable",
    "arguments",
    "output",
    "error",
    "log",
    "queue",
    "initialdir",
}
_CONFIG_KEYS = {
    "request_cpus",
    "request_memory",
    "request_disk",
    "priority",
    "container_image",
    "environment_file",
    "poll_interval",
    "timeout",
    "cancel_on_interrupt",
    "keep_successful_artifacts",
    "extra_submit_attributes",
    "log_path",
}


class HTCondorBackend:
    """Submit one HTCondor process for every job specification."""

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
            raise BackendConfigurationError("request_cpus must be a positive integer.") from exc
        if isinstance(raw_request_cpus, float) and not raw_request_cpus.is_integer():
            raise BackendConfigurationError("request_cpus must be a positive integer.")
        if isinstance(raw_request_cpus, bool):
            raise BackendConfigurationError("request_cpus must be a positive integer.")
        if request_cpus < 1:
            raise BackendConfigurationError("request_cpus must be a positive integer.")

    @staticmethod
    def _validate_extra_attributes(config):
        """Validate custom scheduler attributes."""
        extra_attributes = config.get("extra_submit_attributes", {})
        if not isinstance(extra_attributes, dict):
            raise BackendConfigurationError("extra_submit_attributes must be a mapping.")
        protected = _PROTECTED_ATTRIBUTES & {str(name).lower() for name in extra_attributes}
        if protected:
            names = ", ".join(sorted(protected))
            raise BackendConfigurationError(f"Protected submit attribute(s): {names}.")

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
    def _read_environment_file(path):
        """Convert a simple dotenv file into an HTCondor environment value."""
        if path is None:
            return None
        entries = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            if "=" not in line:
                raise BackendConfigurationError(
                    f"Invalid environment entry in {path}: expected KEY=VALUE."
                )
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if not key:
                raise BackendConfigurationError(f"Invalid empty environment key in {path}.")
            entries.append(f"{key}={value}")
        return " ".join(shlex.quote(entry) for entry in entries)

    def submit(self, job_specs, options):
        """Create and submit a shared-filesystem HTCondor cluster."""
        jobs = self._prepare_jobs(job_specs, options)
        if not jobs:
            return SubmissionHandle(
                backend="htcondor",
                work_dir=Path(options.work_dir or Path.cwd()),
                job_ids=(),
            )
        htcondor = self._load_htcondor()
        config = self._build_config(options)
        self._validate_config(config)
        self._validate_job_resources(jobs)
        work_dir = self._create_work_dir(options)
        self._serialize_jobs(jobs, work_dir)
        event_log = self._resolve_event_log(config, work_dir)
        submit_values, resource_defaults, resource_keys = self._build_submit_values(
            config, jobs, work_dir, event_log
        )
        itemdata = self._build_itemdata(jobs, resource_defaults, resource_keys)
        try:
            submission = htcondor.Submit(submit_values)
            result = self._schedd.submit(submission, count=len(jobs), itemdata=iter(itemdata))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            raise BackendSubmissionError(f"HTCondor submission failed: {exc}") from exc
        handle = self._make_handle(result, jobs, work_dir, event_log, config)
        logger.info("Submitted %d HTCondor jobs as cluster %d", len(jobs), handle.scheduler_id)
        return handle

    @staticmethod
    def _prepare_jobs(job_specs, options):
        """Materialize jobs and attach a configured initializer when needed."""
        jobs = list(job_specs)
        if options.initializer is None:
            return jobs
        return [
            replace(job, initializer=options.initializer, initargs=tuple(options.initargs))
            if job.initializer is None
            else job
            for job in jobs
        ]

    @staticmethod
    def _build_config(options):
        """Merge explicit execution options into backend configuration."""
        config = dict(options.backend_config)
        option_values = {
            "request_cpus": options.request_cpus,
            "request_memory": options.request_memory,
            "request_disk": options.request_disk,
            "priority": options.priority,
            "container_image": options.container_image,
            "environment_file": options.environment_file,
            "poll_interval": options.poll_interval,
            "timeout": options.timeout,
            "cancel_on_interrupt": options.cancel_on_interrupt,
            "keep_successful_artifacts": options.keep_successful_artifacts,
            "extra_submit_attributes": options.extra_submit_attributes,
        }
        for key, value in option_values.items():
            if value not in (None, {}, False) and key not in config:
                config[key] = value
        return config

    @staticmethod
    def _validate_job_resources(jobs):
        """Validate per-job container overrides."""
        for job in jobs:
            image = job.resources.get("container_image")
            if image is not None and not Path(image).is_file():
                raise BackendConfigurationError(
                    f"Configured container_image does not exist: {image}"
                )

    @staticmethod
    def _create_work_dir(options):
        """Create the shared, private run directory."""
        root = Path(options.work_dir or Path.cwd() / "simtools-jobs")
        root.mkdir(parents=True, exist_ok=True)
        work_dir = root / f"htcondor-{uuid.uuid4().hex}"
        work_dir.mkdir(mode=0o700)
        for directory in (
            work_dir / "inputs",
            work_dir / "results",
            work_dir / "stdout",
            work_dir / "stderr",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return work_dir

    @staticmethod
    def _serialize_jobs(jobs, work_dir):
        """Serialize one payload per job."""
        for job in jobs:
            try:
                with (work_dir / "inputs" / f"{job.job_id}.pkl").open("wb") as handle:
                    pickle.dump(job, handle)
            except (OSError, pickle.PickleError, TypeError, AttributeError) as exc:
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

    def _build_submit_values(self, config, jobs, work_dir, event_log):
        """Build the submit description and per-job resource metadata."""
        submit_values = {
            "executable": sys.executable,
            "arguments": shlex.join(
                [
                    "-m",
                    "simtools.job_execution.worker",
                    "--run-directory",
                    str(work_dir),
                    "--job-id",
                    "$(job_id)",
                ]
            ),
            "initialdir": str(work_dir),
            "output": str(work_dir / "stdout" / "$(job_id).out"),
            "error": str(work_dir / "stderr" / "$(job_id).err"),
            "log": str(event_log),
            "should_transfer_files": "NO",
            "request_cpus": str(config.get("request_cpus", 1)),
        }
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
        for key in resource_keys:
            submit_values[key] = f"$({key})"
        for key in ("request_memory", "request_disk", "priority", "container_image"):
            if config.get(key) is not None:
                submit_values[key] = str(config[key])
        if config.get("container_image") or "container_image" in resource_keys:
            submit_values["universe"] = "container"
        environment = self._read_environment_file(config.get("environment_file"))
        if environment:
            submit_values["environment"] = environment
        submit_values.update(config.get("extra_submit_attributes", {}))
        return submit_values, resource_defaults, resource_keys

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
    def _make_handle(result, jobs, work_dir, event_log, config):
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
                "keep_successful_artifacts": bool(config.get("keep_successful_artifacts", False)),
                "request_cpus": config.get("request_cpus", 1),
                "request_memory": config.get("request_memory"),
                "request_disk": config.get("request_disk"),
                "priority": config.get("priority"),
                "container_image": config.get("container_image"),
                "environment_file": str(config["environment_file"])
                if config.get("environment_file")
                else None,
                "event_log": str(event_log),
                "python_version": platform.python_version(),
                "simtools_version": simtools_version,
                "indices": {job.job_id: job.index for job in jobs},
            },
        )

    def wait(self, submission):
        """Wait for all HTCondor processes and load their result files."""
        if not submission.job_ids:
            return []
        failures = self._wait_for_processes(submission)
        results, result_failures = self._load_results(submission)
        failures.extend(result_failures)
        if failures:
            raise BackendExecutionError("HTCondor job failure(s): " + "; ".join(failures))
        self._cleanup_successful_artifacts(submission)
        return sorted(results, key=lambda result: result.index)

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
                break
            event = self._next_event(event_log, submission.metadata["poll_interval"])
            if event is None:
                continue
            event_failures = self._process_event(
                event, submission.scheduler_id, remaining, terminal_types
            )
            failures.extend(event_failures)
        return failures

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
        event_name = str(getattr(event, "type", "")).upper().rsplit(".", maxsplit=1)[-1]
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
        if self._schedd is None or submission.scheduler_id is None:
            return
        try:
            self._schedd.act(
                self._htcondor.JobAction.Remove, f"ClusterId == {submission.scheduler_id}"
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            raise BackendExecutionError(f"Cannot cancel HTCondor cluster: {exc}") from exc
