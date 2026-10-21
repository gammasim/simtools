"""Unit tests for direct child-process resource accounting."""

import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest

from simtools.job_execution import process_accounting

RESOURCE_FILENAME = "resources.json"
STDOUT_FILENAME = "simulation.log.gz"
STDERR_FILENAME = "simulation.err.gz"
SIM_TELARRAY_ROLE = "sim_telarray"
CORSIKA_ROLE = "corsika"
ENCODING = "utf-8"
STDOUT_STREAM = "stdout"
STDERR_STREAM = "stderr"
RETURN_CODE_KEY = "return_code"
SIMULATION_COMMAND = "simulation"
POPEN_ATTRIBUTE = "Popen"
STDERR_LOG_FAILURE = "stderr_log"
DISK_FULL_MESSAGE = "output disk is full"


def test_build_accounting_command_includes_optional_arguments(tmp_test_directory):
    command = process_accounting.build_accounting_command(
        ["simulation executable", "--option"],
        tmp_test_directory / RESOURCE_FILENAME,
        SIM_TELARRAY_ROLE,
        12,
        model_version="7.0.0",
        log_file=tmp_test_directory / STDOUT_FILENAME,
        input_file=tmp_test_directory / "input.eventio",
    )

    assert command[:3] == [sys.executable, "-m", "simtools.job_execution.process_accounting"]
    assert command[-3:] == ["--", "simulation executable", "--option"]
    assert "--model-version" in command
    assert "--log-file" in command
    assert "--input-file" in command


def test_build_accounting_command_without_optional_arguments(tmp_test_directory):
    command = process_accounting.build_accounting_command(
        [SIMULATION_COMMAND], tmp_test_directory / RESOURCE_FILENAME, CORSIKA_ROLE, 1
    )

    assert "--model-version" not in command
    assert "--log-file" not in command
    assert "--input-file" not in command


def test_run_and_record_writes_direct_process_record(tmp_test_directory):
    input_file = tmp_test_directory / "input.txt"
    input_file.write_text("eventio input", encoding=ENCODING)
    output_file = tmp_test_directory / RESOURCE_FILENAME
    log_file = tmp_test_directory / STDOUT_FILENAME

    return_code = process_accounting.run_and_record(
        [sys.executable, "-c", "import sys; print(sys.stdin.read())"],
        output_file,
        CORSIKA_ROLE,
        5,
        model_version="7.0.0",
        log_file=log_file,
        input_file=input_file,
    )

    record = json.loads(output_file.read_text(encoding=ENCODING))
    assert return_code == 0
    assert record["role"] == CORSIKA_ROLE
    assert record["run_id"] == "5"
    assert record["model_version"] == "7.0.0"
    assert record[RETURN_CODE_KEY] == 0
    assert record["signal"] is None
    assert record["wall_time_seconds"] > 0
    assert record["measurement_method"] in {"linux_procfs_direct_pid", "wall_clock_only"}
    with gzip.open(log_file, "rt", encoding=ENCODING) as handle:
        assert handle.read() == "eventio input\n"


def test_run_and_record_records_failing_process(tmp_test_directory):
    output_file = tmp_test_directory / RESOURCE_FILENAME

    return_code = process_accounting.run_and_record(
        [sys.executable, "-c", "raise SystemExit(3)"], output_file, "multipipe", 9
    )

    record = json.loads(output_file.read_text(encoding=ENCODING))
    assert return_code == 3
    assert record[RETURN_CODE_KEY] == 3
    assert record["signal"] is None


def test_run_and_record_accepts_environment_without_log_file(tmp_test_directory):
    output_file = tmp_test_directory / RESOURCE_FILENAME

    return_code = process_accounting.run_and_record(
        [sys.executable, "-c", "import os; assert os.environ['SIMTOOLS_TEST_VALUE'] == 'set'"],
        output_file,
        SIM_TELARRAY_ROLE,
        8,
        env={"SIMTOOLS_TEST_VALUE": "set"},
    )

    assert return_code == 0
    assert json.loads(output_file.read_text(encoding=ENCODING))["role"] == SIM_TELARRAY_ROLE


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
        process_accounting._start_process([SIMULATION_COMMAND], None, input_file, None)


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
        [
            "process_accounting",
            "--output",
            "x",
            "--role",
            CORSIKA_ROLE,
            "--run-id",
            "1",
            "--",
            "run",
        ],
    )
    run_and_record = mocker.patch(
        "simtools.job_execution.process_accounting.run_and_record", return_value=7
    )

    assert process_accounting.main() == 7
    run_and_record.assert_called_once_with(
        ["run"], "x", CORSIKA_ROLE, "1", model_version=None, log_file=None, input_file=None
    )


@pytest.mark.parametrize(
    "streams", [(), (STDOUT_STREAM,), (STDERR_STREAM,), (STDOUT_STREAM, STDERR_STREAM)]
)
def test_separate_logs_retain_large_concurrent_streams(tmp_test_directory, streams):
    log_file = tmp_test_directory / STDOUT_FILENAME
    output_file = tmp_test_directory / RESOURCE_FILENAME
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
        [sys.executable, "-c", command], output_file, SIM_TELARRAY_ROLE, 1, log_file=log_file
    )

    assert return_code == 7
    assert json.loads(output_file.read_text(encoding=ENCODING))[RETURN_CODE_KEY] == 7
    for stream, byte, path in (
        (STDOUT_STREAM, b"O", log_file),
        (STDERR_STREAM, b"E", Path(log_file).with_name(STDERR_FILENAME)),
    ):
        with gzip.open(path, "rb") as handle:
            output = handle.read()
        expected_bytes = 1024 * 1024 if stream in streams else 0
        assert output == byte * expected_bytes


@pytest.mark.parametrize("failure", ["launch", STDERR_LOG_FAILURE])
def test_start_process_closes_capture_files_on_error(tmp_test_directory, mocker, failure):
    opened_files = []
    real_open = gzip.open

    def open_log(path, mode):
        if failure == STDERR_LOG_FAILURE and str(path).endswith(".err.gz"):
            raise OSError("cannot open stderr log")
        handle = real_open(path, mode)
        opened_files.append(handle)
        return handle

    mocker.patch.object(process_accounting.gzip, "open", side_effect=open_log)
    mocker.patch.object(
        process_accounting.subprocess,
        POPEN_ATTRIBUTE,
        side_effect=OSError("cannot launch simulator"),
    )

    with pytest.raises(OSError, match=r"cannot (launch simulator|open stderr log)"):
        process_accounting._start_process(
            [SIMULATION_COMMAND], tmp_test_directory / STDOUT_FILENAME, None, None
        )

    assert opened_files
    assert all(handle.closed for handle in opened_files)


def test_start_process_inherits_streams_without_log_file(mocker):
    start_process = mocker.patch.object(process_accounting.subprocess, POPEN_ATTRIBUTE)

    _, opened_files, log_writers = process_accounting._start_process(
        [SIMULATION_COMMAND], None, None, None
    )

    assert start_process.call_args.kwargs[STDOUT_STREAM] is None
    assert start_process.call_args.kwargs[STDERR_STREAM] is None
    assert not opened_files
    assert not log_writers


def test_run_and_record_preserves_nonstandard_log_filename(tmp_test_directory):
    log_file = tmp_test_directory / "custom.gz"
    return_code = process_accounting.run_and_record(
        [sys.executable, "-c", "import sys; sys.stderr.write('diagnostic')"],
        tmp_test_directory / RESOURCE_FILENAME,
        CORSIKA_ROLE,
        1,
        log_file=log_file,
    )

    assert return_code == 0
    with gzip.open(tmp_test_directory / "custom.gz.err.gz", "rt", encoding=ENCODING) as handle:
        assert handle.read() == "diagnostic"


def test_main_preserves_inherited_binary_stdin_with_separate_logs(tmp_test_directory):
    photon_data = bytes(range(256)) * 1024
    log_file = tmp_test_directory / STDOUT_FILENAME
    output_file = tmp_test_directory / RESOURCE_FILENAME
    stdout_message = b"consumed eventIO photons\n"
    stderr_message = b"simulation diagnostic\n"
    child_command = (
        "import sys; "
        f"assert sys.stdin.buffer.read() == {photon_data[:256]!r} * 1024; "
        f"sys.stdout.buffer.write({stdout_message!r}); "
        f"sys.stderr.buffer.write({stderr_message!r}); "
        "raise SystemExit(7)"
    )
    accounting_command = process_accounting.build_accounting_command(
        [sys.executable, "-c", child_command],
        output_file,
        SIM_TELARRAY_ROLE,
        1,
        log_file=log_file,
    )

    result = subprocess.run(accounting_command, input=photon_data, check=False, timeout=10)

    assert result.returncode == 7
    assert json.loads(output_file.read_text(encoding=ENCODING))[RETURN_CODE_KEY] == 7
    with gzip.open(log_file, "rb") as handle:
        assert handle.read() == stdout_message
    with gzip.open(tmp_test_directory / STDERR_FILENAME, "rb") as handle:
        assert handle.read() == stderr_message


def test_run_and_record_reports_capture_failure_and_reaps_child(tmp_test_directory, mocker):
    log_writers = []
    real_open = gzip.open
    real_start = process_accounting._start_process
    processes = []

    def open_failing_log(path, mode):
        handle = real_open(path, mode)
        mocker.patch.object(handle, "write", side_effect=OSError(DISK_FULL_MESSAGE))
        log_writers.append(handle)
        return handle

    def start_process(*args):
        result = real_start(*args)
        processes.append(result[0])
        return result

    mocker.patch.object(process_accounting.gzip, "open", side_effect=open_failing_log)
    mocker.patch.object(process_accounting, "_start_process", side_effect=start_process)
    child_command = (
        "import os; [(os.write(1, b'O' * 4096), os.write(2, b'E' * 4096)) for _ in range(1024)]"
    )
    with pytest.raises(RuntimeError, match="Failed to capture simulation output") as exc_info:
        process_accounting.run_and_record(
            [sys.executable, "-c", child_command],
            tmp_test_directory / RESOURCE_FILENAME,
            SIM_TELARRAY_ROLE,
            1,
            log_file=tmp_test_directory / STDOUT_FILENAME,
        )

    assert str(exc_info.value.__cause__) == DISK_FULL_MESSAGE
    assert processes[0].returncode is not None
    assert processes[0].stdout.closed
    assert processes[0].stderr.closed
    assert all(handle.closed for handle in log_writers)


def test_copy_process_output_reports_read_failure_after_child_exits(mocker):
    read_error = OSError("cannot read simulator output")
    process_output = mocker.Mock()
    process_output.read.side_effect = read_error
    process = mocker.Mock()
    process.kill.side_effect = ProcessLookupError
    capture_errors = []

    process_accounting._copy_process_output(process_output, mocker.Mock(), process, capture_errors)

    assert capture_errors == [read_error]
