#!/usr/bin/python3

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import resource_benchmark


def _measurement(wall_time, cpu_time=1.0, peak_rss=10.0):
    return {
        "wall_time_s": wall_time,
        "cpu_time_s": cpu_time,
        "peak_rss_mib": peak_rss,
    }


def _metadata():
    return {
        "model_version": "7.0.0",
        "python": "3.14.0",
        "runner_os": "Linux",
        "runner_arch": "X64",
        "runner_image_os": "ubuntu24",
        "runner_image_version": "20260801.1",
        "container_image": "simtools-dev:test",
        "sample_interval_s": 0.2,
    }


def test_measurement_keeps_cpu_from_exited_children():
    measurement = resource_benchmark._Measurement(
        started=1.0,
        cpu_by_process={"pytest": 10.0},
        rss_bytes=10 * 1024 * 1024,
    )
    measurement.update(
        {"pytest": 11.0, "finished-child": 4.0},
        30 * 1024 * 1024,
    )
    measurement.update({"pytest": 12.0}, 20 * 1024 * 1024)

    result = measurement.result(finished=3.0)

    assert result == {
        "wall_time_s": 2.0,
        "cpu_time_s": 6.0,
        "peak_rss_mib": 30.0,
    }


def test_records_publish_only_slow_integration_tests():
    tests = [
        {**_measurement(1.0), "nodeid": "tests/test_app.py::test_fast", "outcome": "passed"},
        {
            **_measurement(8.0),
            "nodeid": "tests/test_app.py::test_applications_from_config[simtools-slow_run]",
            "outcome": "passed",
        },
    ]

    records = resource_benchmark._records(
        "integration", _measurement(10.0), tests, _metadata(), 0, minimum_wall_time=5.0
    )

    assert len(records) == 6
    assert {record["name"].split(" / ")[-2] for record in records} == {
        "7.0.0",
        "simtools-slow_run",
    }
    assert all("python=3.14.0" in record["extra"] for record in records)
    assert all("test_fast" not in record["name"] for record in records)


def test_records_publish_only_unit_suite():
    records = resource_benchmark._records(
        "unit", _measurement(10.0), [], _metadata(), 0, minimum_wall_time=0.0
    )

    assert len(records) == 3
    assert all(record["name"].startswith("unit-suite / ") for record in records)


def test_collection_excludes_reasoned_integration_configuration(mocker, tmp_test_directory):
    config = mocker.MagicMock()
    plugin = resource_benchmark.ResourceBenchmarkPlugin(
        config, Path(tmp_test_directory), "integration", 0.2, 5.0
    )
    included = SimpleNamespace(
        nodeid="test_module.py::test_app[included]",
        callspec=SimpleNamespace(params={"config": {"application": "simtools-included"}}),
    )
    excluded = SimpleNamespace(
        nodeid="test_module.py::test_app[excluded]",
        callspec=SimpleNamespace(
            params={
                "config": {
                    "application": "simtools-excluded",
                    "exclude_from_resource_benchmark": "unstable service",
                }
            }
        ),
    )
    items = [included, excluded]

    plugin.pytest_collection_modifyitems(config, items)

    assert items == [included]
    assert plugin.excluded == [
        {"nodeid": "test_module.py::test_app[excluded]", "reason": "unstable service"}
    ]
    config.hook.pytest_deselected.assert_called_once_with(items=[excluded])


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("resource_benchmark_mode", None, "--resource-benchmark-mode is required"),
        (
            "resource_benchmark_sample_interval",
            0.0,
            "--resource-benchmark-sample-interval must be positive",
        ),
        (
            "resource_benchmark_min_wall_time",
            -1.0,
            "--resource-benchmark-min-wall-time cannot be negative",
        ),
    ],
)
def test_pytest_configure_rejects_invalid_options(mocker, option, value, message):
    options = {
        "resource_benchmark_output": Path("benchmark"),
        "resource_benchmark_mode": "unit",
        "resource_benchmark_sample_interval": 0.2,
        "resource_benchmark_min_wall_time": 5.0,
    }
    options[option] = value
    config = mocker.MagicMock()
    config.getoption.side_effect = options.get

    with pytest.raises(pytest.UsageError, match=message):
        resource_benchmark.pytest_configure(config)


@pytest.mark.xfail(
    importlib.util.find_spec("psutil") is None,
    reason="psutil is required by the resource benchmark sampler",
    strict=True,
)
def test_resource_benchmark_plugin_end_to_end(tmp_test_directory, simtools_root_path):
    temporary_path = Path(tmp_test_directory)
    test_file = temporary_path / "test_benchmark_sample.py"
    output = temporary_path / "benchmark"
    test_file.write_text(
        """
import subprocess
import sys
import time

import pytest


def test_child_cpu_is_retained():
    subprocess.run(
        [sys.executable, "-c", "sum(value * value for value in range(30000000))"],
        check=True,
    )
    time.sleep(0.05)


@pytest.mark.parametrize(
    "config",
    [{"exclude_from_resource_benchmark": "not representative"}],
)
def test_excluded(config):
    raise AssertionError("This test must be deselected")
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "resource_benchmark",
            "--override-ini=addopts=",
            "--resource-benchmark-output",
            str(output),
            "--resource-benchmark-mode",
            "integration",
            "--resource-benchmark-min-wall-time",
            "0",
            "--resource-benchmark-sample-interval",
            "0.01",
            str(test_file),
        ],
        capture_output=True,
        cwd=simtools_root_path,
        env={**os.environ, "PYTHONPATH": str(simtools_root_path / "tests")},
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    raw = json.loads((output / "benchmark-raw.json").read_text(encoding="utf-8"))
    records = json.loads((output / "benchmark-results.json").read_text(encoding="utf-8"))
    assert len(raw["tests"]) == 1
    assert raw["tests"][0]["cpu_time_s"] > 0.1
    assert raw["excluded"] == [
        {
            "nodeid": f"{test_file.name}::test_excluded[config0]",
            "reason": "not representative",
        }
    ]
    assert raw["published_test_count"] == 1
    assert len(records) == 6
