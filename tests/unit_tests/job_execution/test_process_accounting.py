"""Unit tests for direct child-process resource accounting."""

import gzip
import json
import sys
from pathlib import Path

import pytest

from simtools.job_execution import process_accounting


def test_build_accounting_command_includes_optional_arguments(tmp_test_directory):
    command = process_accounting.build_accounting_command(
        ["simulation executable", "--option"],
        tmp_test_directory / "resources.json",
        "sim_telarray",
        12,
        model_version="7.0.0",
        log_file=tmp_test_directory / "simulation.log.gz",
        input_file=tmp_test_directory / "input.eventio",
    )

    assert command[:3] == [sys.executable, "-m", "simtools.job_execution.process_accounting"]
    assert command[-3:] == ["--", "simulation executable", "--option"]
    assert "--model-version" in command
    assert "--log-file" in command
    assert "--input-file" in command


def test_build_accounting_command_without_optional_arguments(tmp_test_directory):
    command = process_accounting.build_accounting_command(
        ["simulation"], tmp_test_directory / "resources.json", "corsika", 1
    )

    assert "--model-version" not in command
    assert "--log-file" not in command
    assert "--input-file" not in command


def test_run_and_record_writes_direct_process_record(tmp_test_directory):
    input_file = tmp_test_directory / "input.txt"
    input_file.write_text("eventio input", encoding="utf-8")
    output_file = tmp_test_directory / "resources.json"
    log_file = tmp_test_directory / "simulation.log.gz"

    return_code = process_accounting.run_and_record(
        [sys.executable, "-c", "import sys; print(sys.stdin.read())"],
        output_file,
        "corsika",
        5,
        model_version="7.0.0",
        log_file=log_file,
        input_file=input_file,
    )

    record = json.loads(output_file.read_text(encoding="utf-8"))
    assert return_code == 0
    assert record["role"] == "corsika"
    assert record["run_id"] == "5"
    assert record["model_version"] == "7.0.0"
    assert record["return_code"] == 0
    assert record["signal"] is None
    assert record["wall_time_seconds"] > 0
    assert record["measurement_method"] in {"linux_procfs_direct_pid", "wall_clock_only"}
    with gzip.open(log_file, "rt", encoding="utf-8") as handle:
        assert handle.read() == "eventio input\n"


def test_run_and_record_records_failing_process(tmp_test_directory):
    output_file = tmp_test_directory / "resources.json"

    return_code = process_accounting.run_and_record(
        [sys.executable, "-c", "raise SystemExit(3)"], output_file, "multipipe", 9
    )

    record = json.loads(output_file.read_text(encoding="utf-8"))
    assert return_code == 3
    assert record["return_code"] == 3
    assert record["signal"] is None


def test_run_and_record_accepts_environment_without_log_file(tmp_test_directory):
    output_file = tmp_test_directory / "resources.json"

    return_code = process_accounting.run_and_record(
        [sys.executable, "-c", "import os; assert os.environ['SIMTOOLS_TEST_VALUE'] == 'set'"],
        output_file,
        "sim_telarray",
        8,
        env={"SIMTOOLS_TEST_VALUE": "set"},
    )

    assert return_code == 0
    assert json.loads(output_file.read_text(encoding="utf-8"))["role"] == "sim_telarray"


def test_read_linux_process_sample_handles_unavailable_process(mocker):
    mocker.patch("simtools.job_execution.process_accounting.platform.system", return_value="Linux")
    mocker.patch.object(Path, "read_text", side_effect=FileNotFoundError)

    assert process_accounting._read_linux_process_sample(12) is None


def test_read_linux_process_sample_skips_non_linux_platform(mocker):
    mocker.patch("simtools.job_execution.process_accounting.platform.system", return_value="Darwin")

    assert process_accounting._read_linux_process_sample(12) is None


def test_read_peak_rss_bytes_prefers_peak_value():
    status = "VmRSS:\t12 kB\nVmHWM:\t25 kB\n"

    assert process_accounting._read_peak_rss_bytes(status) == 25 * 1024


def test_read_peak_rss_bytes_returns_none_without_memory_data():
    assert process_accounting._read_peak_rss_bytes("Name:\tsimulation\n") is None


def test_start_process_closes_opened_files_on_error(tmp_test_directory, mocker):
    input_file = Path(tmp_test_directory) / "input.eventio"
    input_file.write_bytes(b"eventio")
    mocker.patch(
        "simtools.job_execution.process_accounting.subprocess.Popen",
        side_effect=FileNotFoundError("simulation not found"),
    )

    with pytest.raises(FileNotFoundError, match="simulation not found"):
        process_accounting._start_process(["simulation"], None, input_file, None)


def test_main_requires_executable(mocker):
    mocker.patch.object(
        sys, "argv", ["process_accounting", "--output", "x", "--role", "x", "--run-id", "1"]
    )

    with pytest.raises(SystemExit, match="2"):
        process_accounting.main()


def test_main_runs_requested_command(mocker):
    mocker.patch.object(
        sys,
        "argv",
        ["process_accounting", "--output", "x", "--role", "corsika", "--run-id", "1", "--", "run"],
    )
    run_and_record = mocker.patch(
        "simtools.job_execution.process_accounting.run_and_record", return_value=7
    )

    assert process_accounting.main() == 7
    run_and_record.assert_called_once_with(
        ["run"], "x", "corsika", "1", model_version=None, log_file=None, input_file=None
    )


@pytest.mark.parametrize("streams", [("stdout",), ("stderr",), ("stdout", "stderr")])
def test_combined_log_retains_large_concurrent_streams(tmp_test_directory, streams):
    log_file = tmp_test_directory / "simulation.log.gz"
    output_file = tmp_test_directory / "resources.json"
    command = (
        "import os, threading; "
        "threads = [threading.Thread(target=lambda fd=fd, byte=byte: "
        "[os.write(fd, byte * 4096) for _ in range(256)]) "
        f"for fd, byte in [(1, b'O'), (2, b'E')] if "
        f"{{1: 'stdout', 2: 'stderr'}}[fd] in {streams!r}]; "
        "[thread.start() for thread in threads]; "
        "[thread.join() for thread in threads]; raise SystemExit(7)"
    )

    return_code = process_accounting.run_and_record(
        [sys.executable, "-c", command], output_file, "sim_telarray", 1, log_file=log_file
    )

    assert return_code == 7
    assert json.loads(output_file.read_text(encoding="utf-8"))["return_code"] == 7
    with gzip.open(log_file, "rb") as handle:
        output = handle.read()
    for stream, byte in (("stdout", b"O"), ("stderr", b"E")):
        expected_bytes = 1024 * 1024 if stream in streams else 0
        assert output.count(byte) == expected_bytes
    assert len(output) == len(streams) * 1024 * 1024
