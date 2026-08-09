#!/usr/bin/python3

import json
import sys

import pytest

import simtools.testing.resource_benchmark as resource_benchmark
from simtools.testing.resource_benchmark import collect


def test_collect_records_successful_command(tmp_test_directory):
    output = tmp_test_directory / "resource.json"

    result = collect([sys.executable, "-c", "print('ok')"], output, sample_interval=0.01)

    assert result["returncode"] == 0
    assert result["wall_time_s"] > 0
    assert result["cpu_time_s"] >= 0
    assert result["peak_rss_mib"] > 0
    assert json.loads(output.read_text(encoding="utf-8")) == result


def test_collect_records_failing_command(tmp_test_directory):
    output = tmp_test_directory / "resource.json"

    result = collect([sys.executable, "-c", "raise SystemExit(7)"], output, sample_interval=0.01)

    assert result["returncode"] == 7
    assert output.check(file=1)


def test_collect_includes_memory_using_child(tmp_test_directory):
    output = tmp_test_directory / "resource.json"
    child = "import time; data = bytearray(8 * 1024 * 1024); time.sleep(0.15)"

    result = collect([sys.executable, "-c", child], output, sample_interval=0.01)

    assert result["returncode"] == 0
    assert result["peak_rss_mib"] >= 8


@pytest.mark.parametrize("sample_interval", [0, -1])
def test_collect_rejects_invalid_sample_interval(tmp_test_directory, sample_interval):
    with pytest.raises(ValueError, match="sample interval must be positive"):
        collect(
            [sys.executable, "-c", "pass"], tmp_test_directory / "resource.json", sample_interval
        )


def test_collect_rejects_empty_command(tmp_test_directory):
    with pytest.raises(ValueError, match="A command is required"):
        collect([], tmp_test_directory / "resource.json")


def test_main_collects_command_after_separator(monkeypatch, tmp_test_directory, mocker):
    output = tmp_test_directory / "resource.json"
    collected = {"returncode": 4}
    collect_mock = mocker.patch(
        "simtools.testing.resource_benchmark.collect", return_value=collected
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["simtools-test-resources-collect", "--output", str(output), "--", "pytest", "-q"],
    )

    with pytest.raises(SystemExit, match="4"):
        resource_benchmark.main()

    collect_mock.assert_called_once_with(["pytest", "-q"], output, sample_interval=0.2)


def test_records_main_writes_benchmark_records(monkeypatch, tmp_test_directory):
    input_path = tmp_test_directory / "measurement.json"
    output_path = tmp_test_directory / "records.json"
    input_path.write_text(
        json.dumps(
            {
                "returncode": 0,
                "wall_time_s": 1.0,
                "cpu_time_s": 0.5,
                "peak_rss_mib": 12.0,
                "cpu_utilisation": 50.0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simtools-test-resources-make-benchmark-records",
            "--input",
            str(input_path),
            "--name",
            "unit-suite",
            "--output",
            str(output_path),
        ],
    )

    resource_benchmark.records_main()

    records = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(records) == 4
    assert records[-1] == {
        "name": "unit-suite / cpu_utilisation",
        "unit": "percent",
        "value": 50.0,
        "extra": "exit_code=0",
    }


def test_records_main_rejects_mismatched_inputs(monkeypatch, tmp_test_directory):
    input_path = tmp_test_directory / "measurement.json"
    input_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simtools-test-resources-make-benchmark-records",
            "--input",
            str(input_path),
            "--name",
            "one",
            "--name",
            "two",
            "--output",
            str(tmp_test_directory / "records.json"),
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        resource_benchmark.records_main()


def test_resource_benchmark_test_ids_excludes_configurations():
    included = {"application": "simtools-a", "test_name": "run"}
    excluded = {
        "application": "simtools-b",
        "test_name": "run",
        "exclude_from_resource_benchmark": "slow",
    }

    assert resource_benchmark._resource_benchmark_test_ids([included, excluded]) == [
        "simtools-a_run"
    ]


def test_integration_main_writes_records_and_metadata(monkeypatch, mocker, tmp_test_directory):
    configs = [{"application": "simtools-example", "test_name": "run"}]
    monkeypatch.setattr(
        resource_benchmark,
        "get_list_of_test_configurations",
        mocker.Mock(return_value=(configs, ["simtools-example_run"])),
    )
    monkeypatch.setattr(
        resource_benchmark,
        "get_resource_benchmark_configurations",
        mocker.Mock(return_value=(configs, [])),
    )

    def fake_collect(command, output, sample_interval=0.2):
        measurement = {
            "command": command,
            "returncode": 0,
            "wall_time_s": 1.0,
            "cpu_time_s": 0.5,
            "peak_rss_mib": 10.0,
            "cpu_utilisation": 50.0,
        }
        output.write_text(json.dumps(measurement), encoding="utf-8")
        return measurement

    monkeypatch.setattr(resource_benchmark, "collect", fake_collect)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simtools-test-resources-run-integration-benchmark",
            "--model-version",
            "7.0.0",
            "--output-dir",
            str(tmp_test_directory),
            "--simulation-models-path",
            "models",
            "--test-resources-path",
            "resources",
        ],
    )
    monkeypatch.setenv("GITHUB_SHA", "abc123")

    resource_benchmark.integration_main()

    records = json.loads(
        (tmp_test_directory / "benchmark-results.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (tmp_test_directory / "benchmark-metadata.json").read_text(encoding="utf-8")
    )
    assert len(records) == 8
    assert metadata["model_version"] == "7.0.0"
    assert metadata["commit"] == "abc123"
    assert metadata["included_test_ids"] == ["simtools-example_run"]


def test_integration_main_stops_after_failed_test(monkeypatch, mocker, tmp_test_directory):
    configs = [{"application": "simtools-example", "test_name": "run"}]
    mocker.patch(
        "simtools.testing.resource_benchmark.get_list_of_test_configurations",
        return_value=(configs, ["simtools-example_run"]),
    )
    mocker.patch(
        "simtools.testing.resource_benchmark.get_resource_benchmark_configurations",
        return_value=(configs, []),
    )
    measurements = iter([0, 7])

    def fake_collect(command, output, sample_interval=0.2):
        returncode = next(measurements)
        measurement = {
            "command": command,
            "returncode": returncode,
            "wall_time_s": 1.0,
            "cpu_time_s": 0.5,
            "peak_rss_mib": 10.0,
            "cpu_utilisation": 50.0,
        }
        output.write_text(json.dumps(measurement), encoding="utf-8")
        return measurement

    monkeypatch.setattr(resource_benchmark, "collect", fake_collect)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simtools-test-resources-run-integration-benchmark",
            "--model-version",
            "7.0.0",
            "--output-dir",
            str(tmp_test_directory),
        ],
    )

    with pytest.raises(SystemExit, match="1"):
        resource_benchmark.integration_main()


def test_integration_main_requires_included_tests(monkeypatch, mocker, tmp_test_directory):
    mocker.patch(
        "simtools.testing.resource_benchmark.get_list_of_test_configurations",
        return_value=([], []),
    )
    mocker.patch(
        "simtools.testing.resource_benchmark.get_resource_benchmark_configurations",
        return_value=([], []),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simtools-test-resources-run-integration-benchmark",
            "--model-version",
            "7.0.0",
            "--output-dir",
            str(tmp_test_directory),
        ],
    )

    with pytest.raises(SystemExit, match="No integration tests remain"):
        resource_benchmark.integration_main()
