"""Collect resource usage for a pytest command and its child processes."""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import psutil

from simtools.testing.configuration import (
    get_list_of_test_configurations,
    get_resource_benchmark_configurations,
)

_BYTES_PER_MIB = 1024 * 1024
_BENCHMARK_METRICS = ("wall_time_s", "cpu_time_s", "peak_rss_mib", "cpu_utilisation")
_TEST_MODULE = "tests/integration_tests/test_applications_from_config.py"
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def _process_snapshot(process):
    """Return CPU seconds and RSS in bytes for a process tree."""
    processes = [process]
    try:
        processes.extend(process.children(recursive=True))
    except psutil.NoSuchProcess, psutil.AccessDenied, OSError:
        pass

    cpu_seconds = 0.0
    rss_bytes = 0
    for child in processes:
        try:
            cpu_times = child.cpu_times()
            memory = child.memory_info()
        except psutil.NoSuchProcess, psutil.AccessDenied, OSError:
            continue
        cpu_seconds += cpu_times.user + cpu_times.system
        rss_bytes += memory.rss
    return cpu_seconds, rss_bytes


def collect(command, output, sample_interval=0.2):
    """Run a command and write process-tree resource measurements.

    Parameters
    ----------
    command: list[str]
        Executable and arguments to run.
    output: str or pathlib.Path
        JSON file to write. The file is replaced atomically after collection.
    sample_interval: float, optional
        Seconds between process-tree samples.

    Returns
    -------
    dict
        Measurements and the command exit code.

    Raises
    ------
    ValueError
        If the command is empty or the sample interval is not positive.
    """
    if not command:
        raise ValueError("A command is required for resource collection")
    if sample_interval <= 0:
        raise ValueError("The sample interval must be positive")

    started = time.monotonic()
    with subprocess.Popen(command) as process:
        process_info = psutil.Process(process.pid)
        peak_rss = 0
        cpu_seconds = 0.0

        cpu_seconds, rss_bytes = _process_snapshot(process_info)
        peak_rss = max(peak_rss, rss_bytes)
        while process.poll() is None:
            cpu_seconds, rss_bytes = _process_snapshot(process_info)
            peak_rss = max(peak_rss, rss_bytes)
            time.sleep(sample_interval)

        final_cpu, final_rss = _process_snapshot(process_info)
        cpu_seconds = max(cpu_seconds, final_cpu)
        peak_rss = max(peak_rss, final_rss)
    result = {
        "command": command,
        "returncode": process.returncode,
        "wall_time_s": time.monotonic() - started,
        "cpu_time_s": cpu_seconds,
        "peak_rss_mib": peak_rss / _BYTES_PER_MIB,
        "sample_interval_s": sample_interval,
    }
    wall_time = result["wall_time_s"]
    result["cpu_utilisation"] = 100 * cpu_seconds / wall_time if wall_time else 0.0
    _write_json_atomically(result, Path(output))
    return result


def _write_json_atomically(data, output):
    """Write JSON next to the target and atomically replace the target."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    """Run the resource collector command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-interval", type=float, default=0.2)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    result = collect(command, args.output, sample_interval=args.sample_interval)
    raise SystemExit(result["returncode"])


def records_for_measurement(name, measurement):
    """Return github-action-benchmark records for one measurement."""
    return [
        {
            "name": f"{name} / {metric}",
            "unit": "percent" if metric == "cpu_utilisation" else metric,
            "value": measurement[metric],
            "extra": f"exit_code={measurement['returncode']}",
        }
        for metric in _BENCHMARK_METRICS
    ]


def records_main():
    """Convert collector JSON files to custom benchmark records."""
    parser = argparse.ArgumentParser(description=records_main.__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--name", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(args.input) != len(args.name):
        parser.error("--input and --name must be supplied the same number of times")

    records = []
    for input_path, name in zip(args.input, args.name):
        records.extend(records_for_measurement(name, json.loads(input_path.read_text())))
    args.output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


def _measurement_name(model_version, test_id):
    """Return a stable dashboard name for an integration test."""
    return f"integration / {model_version} / {test_id}"


def _pytest_command(
    model_version, simulation_models_path=None, test_resources_path=None, node_id=None
):
    """Build the serial pytest command used for one measurement."""
    command = [
        sys.executable,
        "-m",
        "pytest",
        f"--model_version={model_version}",
        "--color=yes",
        "--no-cov",
        "-n",
        "0",
        _TEST_MODULE,
    ]
    if simulation_models_path:
        command.insert(-1, f"--simulation-models-path={simulation_models_path}")
    if test_resources_path:
        command.insert(-1, f"--test-resources-path={test_resources_path}")
    if node_id:
        command[-1] = f"{_TEST_MODULE}::test_applications_from_config[{node_id}]"
    return command


def _resource_benchmark_test_ids(configs):
    """Return pytest IDs while preserving duplicate parametrization suffixes."""
    included, _ = get_resource_benchmark_configurations(configs)
    included_objects = {id(config) for config in included}
    base_ids = [
        f"{config.get('application', 'no-app-name')}_{config.get('test_name', 'no-test-name')}"
        for config in configs
    ]
    counts = Counter(base_ids)
    occurrences = defaultdict(int)
    test_ids = []
    for config, base_id in zip(configs, base_ids):
        occurrence = occurrences[base_id]
        occurrences[base_id] += 1
        test_id = base_id if counts[base_id] == 1 else f"{base_id}{occurrence}"
        if id(config) in included_objects:
            test_ids.append(test_id)
    return test_ids


def _run_integration_measurement(command, name, output_dir, records):
    """Collect one integration command and append benchmark records."""
    raw_path = output_dir / f"{_SAFE_NAME.sub('_', name)}.json"
    measurement = collect(command, raw_path)
    records.extend(records_for_measurement(name, measurement))
    return measurement


def integration_main():
    """Run serial resource measurements for integration test nodes."""
    parser = argparse.ArgumentParser(description=integration_main.__doc__)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--simulation-models-path")
    parser.add_argument("--test-resources-path")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config_files = sorted(Path("tests/integration_tests/config").glob("*.yml"))
    configs, _ = get_list_of_test_configurations(
        config_files, test_resources_path=args.test_resources_path
    )
    _, excluded = get_resource_benchmark_configurations(configs)
    test_ids = _resource_benchmark_test_ids(configs)
    if not test_ids:
        raise SystemExit("No integration tests remain after resource benchmark exclusions")

    records = []
    suite_name = f"integration-suite / {args.model_version}"
    suite_measurement = _run_integration_measurement(
        _pytest_command(args.model_version, args.simulation_models_path, args.test_resources_path),
        suite_name,
        args.output_dir,
        records,
    )
    for test_id in test_ids:
        measurement = _run_integration_measurement(
            _pytest_command(
                args.model_version,
                args.simulation_models_path,
                args.test_resources_path,
                test_id,
            ),
            _measurement_name(args.model_version, test_id),
            args.output_dir,
            records,
        )
        if measurement["returncode"] != 0:
            break

    (args.output_dir / "benchmark-results.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "benchmark-metadata.json").write_text(
        json.dumps(
            {
                "model_version": args.model_version,
                "commit": os.environ.get("GITHUB_SHA"),
                "run_id": os.environ.get("GITHUB_RUN_ID"),
                "runner_os": os.environ.get("RUNNER_OS"),
                "runner_arch": os.environ.get("RUNNER_ARCH"),
                "container_image": os.environ.get("CONTAINER_IMAGE"),
                "collector_version": "1",
                "included_test_ids": test_ids,
                "excluded": excluded,
                "suite_returncode": suite_measurement["returncode"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if suite_measurement["returncode"] != 0 or any(
        record["extra"] != "exit_code=0" for record in records
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
