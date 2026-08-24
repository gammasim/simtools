"""Measure pytest process-tree resources and write benchmark records."""

import json
import os
import platform
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path

import pytest

try:
    import psutil
except ModuleNotFoundError:
    psutil = None

_BYTES_PER_MIB = 1024 * 1024
_METRICS = ("wall_time_s", "cpu_time_s", "peak_rss_mib")
_PLUGIN_NAME = "simtools-resource-benchmark"


def pytest_addoption(parser):
    """Add resource benchmark options."""
    group = parser.getgroup("resource benchmark")
    group.addoption(
        "--resource-benchmark-output",
        type=Path,
        help="Directory for benchmark-results.json and benchmark-raw.json.",
    )
    group.addoption(
        "--resource-benchmark-mode",
        choices=("unit", "integration"),
        help="Record the session only, or the session and integration tests.",
    )
    group.addoption(
        "--resource-benchmark-min-wall-time",
        type=float,
        default=5.0,
        help="Minimum wall time in seconds for publishing an integration-test chart.",
    )
    group.addoption(
        "--resource-benchmark-sample-interval",
        type=float,
        default=0.2,
        help="Seconds between process-tree samples.",
    )


def pytest_configure(config):
    """Register the benchmark plugin when an output directory is requested."""
    output = config.getoption("resource_benchmark_output")
    if output is None:
        return
    if psutil is None:
        raise pytest.UsageError(
            "--resource-benchmark-output requires the optional psutil dependency"
        )
    mode = config.getoption("resource_benchmark_mode")
    if mode is None:
        raise pytest.UsageError("--resource-benchmark-mode is required")
    sample_interval = config.getoption("resource_benchmark_sample_interval")
    minimum_wall_time = config.getoption("resource_benchmark_min_wall_time")
    if sample_interval <= 0:
        raise pytest.UsageError("--resource-benchmark-sample-interval must be positive")
    if minimum_wall_time < 0:
        raise pytest.UsageError("--resource-benchmark-min-wall-time cannot be negative")
    config.pluginmanager.register(
        ResourceBenchmarkPlugin(config, output, mode, sample_interval, minimum_wall_time),
        _PLUGIN_NAME,
    )


class _Measurement:
    """Accumulate process-tree samples without losing exited child CPU time."""

    def __init__(self, started, cpu_by_process, rss_bytes):
        self.started = started
        self.initial_cpu = dict(cpu_by_process)
        self.maximum_cpu = dict(cpu_by_process)
        self.peak_rss = rss_bytes

    def update(self, cpu_by_process, rss_bytes):
        """Add one process-tree sample."""
        for process_id, cpu_seconds in cpu_by_process.items():
            self.maximum_cpu[process_id] = max(cpu_seconds, self.maximum_cpu.get(process_id, 0.0))
        self.peak_rss = max(self.peak_rss, rss_bytes)

    def result(self, finished):
        """Return wall time, cumulative CPU time, and peak RSS."""
        wall_time = max(finished - self.started, 0.0)
        cpu_time = sum(
            maximum - self.initial_cpu.get(process_id, 0.0)
            for process_id, maximum in self.maximum_cpu.items()
        )
        return {
            "wall_time_s": wall_time,
            "cpu_time_s": cpu_time,
            "peak_rss_mib": self.peak_rss / _BYTES_PER_MIB,
        }


class _ResourceSampler:
    """Sample the current pytest process and all descendants in a thread."""

    def __init__(self, sample_interval):
        self.psutil = psutil
        self.process = psutil.Process()
        self.sample_interval = sample_interval
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample_until_stopped, daemon=True)
        initial = self._snapshot()
        self.session = _Measurement(time.monotonic(), *initial)
        self.active_test = None

    def _snapshot(self):
        processes = [self.process]
        try:
            processes.extend(self.process.children(recursive=True))
        except self.psutil.NoSuchProcess, self.psutil.AccessDenied, OSError, TypeError:
            pass

        cpu_by_process = {}
        rss_bytes = 0
        for process in processes:
            try:
                identity = (process.pid, process.create_time())
                cpu_times = process.cpu_times()
                memory = process.memory_info()
            except self.psutil.NoSuchProcess, self.psutil.AccessDenied, OSError, TypeError:
                continue
            cpu_by_process[identity] = cpu_times.user + cpu_times.system
            rss_bytes += memory.rss
        return cpu_by_process, rss_bytes

    def _record(self, snapshot):
        with self.lock:
            self.session.update(*snapshot)
            if self.active_test is not None:
                self.active_test.update(*snapshot)

    def _sample_until_stopped(self):
        while not self.stop_event.wait(self.sample_interval):
            self._record(self._snapshot())

    def start(self):
        """Start background process-tree sampling."""
        self.thread.start()

    def start_test(self):
        """Start a per-test measurement."""
        snapshot = self._snapshot()
        self._record(snapshot)
        with self.lock:
            self.active_test = _Measurement(time.monotonic(), *snapshot)

    def finish_test(self):
        """Finish and return the active per-test measurement."""
        snapshot = self._snapshot()
        finished = time.monotonic()
        with self.lock:
            self.session.update(*snapshot)
            self.active_test.update(*snapshot)
            result = self.active_test.result(finished)
            self.active_test = None
        return result

    def finish(self):
        """Stop sampling and return the session measurement."""
        self.stop_event.set()
        self.thread.join()
        snapshot = self._snapshot()
        finished = time.monotonic()
        with self.lock:
            self.session.update(*snapshot)
            return self.session.result(finished)


class ResourceBenchmarkPlugin:
    """Collect session and optional per-integration-test resource measurements."""

    def __init__(self, config, output, mode, sample_interval, minimum_wall_time):
        self.config = config
        self.output = output
        self.mode = mode
        self.sample_interval = sample_interval
        self.minimum_wall_time = minimum_wall_time
        self.sampler = None
        self.tests = []
        self.outcomes = {}
        self.excluded = []

    def pytest_sessionstart(self):
        """Start session resource sampling."""
        self.sampler = _ResourceSampler(self.sample_interval)
        self.sampler.start()

    def pytest_collection_modifyitems(self, config, items):
        """Remove explicitly excluded integration configurations."""
        if self.mode != "integration":
            return
        selected = []
        deselected = []
        for item in items:
            callspec = getattr(item, "callspec", None)
            workflow = callspec.params.get("config") if callspec is not None else None
            reason = (
                workflow.get("exclude_from_resource_benchmark")
                if isinstance(workflow, Mapping)
                else None
            )
            if reason:
                self.excluded.append({"nodeid": item.nodeid, "reason": reason})
                deselected.append(item)
            else:
                selected.append(item)
        items[:] = selected
        if deselected:
            config.hook.pytest_deselected(items=deselected)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self, item):
        """Measure one complete integration-test protocol."""
        if self.mode != "integration":
            yield
            return
        self.sampler.start_test()
        try:
            yield
        finally:
            measurement = self.sampler.finish_test()
            measurement["nodeid"] = item.nodeid
            measurement["outcome"] = self.outcomes.get(item.nodeid, "passed")
            self.tests.append(measurement)

    def pytest_runtest_logreport(self, report):
        """Remember the final outcome reported for each integration test."""
        if report.failed:
            self.outcomes[report.nodeid] = "failed"
        elif report.skipped and self.outcomes.get(report.nodeid) != "failed":
            self.outcomes[report.nodeid] = "skipped"
        elif report.when == "call":
            self.outcomes.setdefault(report.nodeid, "passed")

    def pytest_sessionfinish(self, exitstatus):
        """Finish sampling and write raw and chart-ready JSON files."""
        session = self.sampler.finish()
        self.sampler = None
        metadata = _metadata(self.config, self.mode, self.sample_interval, self.minimum_wall_time)
        records = _records(
            self.mode,
            session,
            self.tests,
            metadata,
            exitstatus,
            self.minimum_wall_time,
        )
        raw = {
            "schema_version": 1,
            "metadata": metadata,
            "session": {**session, "exit_status": int(exitstatus)},
            "tests": self.tests,
            "excluded": self.excluded,
            "published_test_count": sum(
                test["wall_time_s"] >= self.minimum_wall_time for test in self.tests
            ),
        }
        _write_json(self.output / "benchmark-results.json", records)
        _write_json(self.output / "benchmark-raw.json", raw)

    def pytest_unconfigure(self):
        """Stop a sampler if pytest exits before session finish."""
        if self.sampler is not None:
            self.sampler.finish()
            self.sampler = None


def _metadata(config, mode, sample_interval, minimum_wall_time):
    """Return persistent benchmark environment metadata."""
    model_version = config.getoption("model_version", default=None)
    return {
        "mode": mode,
        "model_version": model_version,
        "commit": os.environ.get("GITHUB_SHA"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "runner_os": os.environ.get("RUNNER_OS"),
        "runner_arch": os.environ.get("RUNNER_ARCH"),
        "runner_image_os": os.environ.get("ImageOS"),
        "runner_image_version": os.environ.get("ImageVersion"),
        "container_image": os.environ.get("CONTAINER_IMAGE"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "sample_interval_s": sample_interval,
        "minimum_wall_time_s": minimum_wall_time,
        "collector_version": 2,
    }


def _metadata_text(metadata, outcome):
    """Return compact metadata stored with every long-term data point."""
    values = {
        "outcome": outcome,
        "python": metadata["python"],
        "runner": f"{metadata['runner_os']}/{metadata['runner_arch']}",
        "runner_image": f"{metadata['runner_image_os']}/{metadata['runner_image_version']}",
        "container": metadata["container_image"],
        "sample_interval_s": metadata["sample_interval_s"],
    }
    return " | ".join(f"{key}={value}" for key, value in values.items())


def _measurement_records(name, measurement, extra):
    """Return github-action-benchmark records for one measurement."""
    return [
        {
            "name": f"{name} / {metric}",
            "unit": metric,
            "value": measurement[metric],
            "extra": extra,
        }
        for metric in _METRICS
    ]


def _short_test_name(nodeid):
    """Return the parameter ID or test name from a pytest node ID."""
    test_name = nodeid.rsplit("::", maxsplit=1)[-1]
    prefix = "test_applications_from_config["
    return test_name[len(prefix) : -1] if test_name.startswith(prefix) else test_name


def _records(mode, session, tests, metadata, exitstatus, minimum_wall_time):
    """Build session records and slow integration-test records."""
    model_version = metadata["model_version"]
    session_name = "unit-session" if mode == "unit" else f"integration-session / {model_version}"
    records = _measurement_records(
        session_name, session, _metadata_text(metadata, f"exit_status={int(exitstatus)}")
    )
    if mode == "unit":
        return records
    for test in tests:
        if test["wall_time_s"] < minimum_wall_time:
            continue
        name = f"integration / {model_version} / {_short_test_name(test['nodeid'])}"
        records.extend(_measurement_records(name, test, _metadata_text(metadata, test["outcome"])))
    return records


def _write_json(path, data):
    """Write formatted JSON, creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
