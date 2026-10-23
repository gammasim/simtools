"""Run one process while recording its directly attributable resource use."""

import argparse
import gzip
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
import zlib
from datetime import UTC, datetime
from pathlib import Path

from simtools.version import __version__

_STDOUT_LOG_SUFFIX = ".log.gz"
_STDERR_LOG_SUFFIX = ".err.gz"
_ENCODING = "utf-8"
_RECORD_SCHEMA_VERSION = "1.1.0"
_CONTAINER_ENVIRONMENT_VARIABLES = ("APPTAINER_CONTAINER", "SINGULARITY_CONTAINER")
_CPUINFO_PATH = Path("/proc/cpuinfo")


def build_accounting_command(
    command,
    output_path,
    role,
    run_id,
    model_version=None,
    log_file=None,
    input_file=None,
):
    """Build a command that records resource use for one executable.

    The returned command preserves the executable's stdin unless ``input_file``
    is supplied. This is important for sim_telarray consumers of an eventIO
    stream.

    Parameters
    ----------
    command : list[str | pathlib.Path]
        Executable and arguments to run.
    output_path : str or pathlib.Path
        JSON resource-record output path.
    role : str
        Process role, such as ``"corsika"`` or ``"sim_telarray"``.
    run_id : int or str
        Production run identifier.
    model_version : str, optional
        Simulation-model version associated with this process.
    log_file : str or pathlib.Path, optional
        Gzip-compressed stdout log file. Stderr is written to a sibling
        ``.err.gz`` file, replacing the ``.log.gz`` suffix.
    input_file : str or pathlib.Path, optional
        File connected to the child's stdin.

    Returns
    -------
    list[str]
        Command invoking this module with the supplied process command.
    """
    accounting_command = [
        sys.executable,
        "-m",
        "simtools.job_execution.process_accounting",
        "--output",
        str(output_path),
        "--role",
        role,
        "--run-id",
        str(run_id),
    ]
    if model_version:
        accounting_command.extend(["--model-version", str(model_version)])
    if log_file:
        accounting_command.extend(["--log-file", str(log_file)])
    if input_file:
        accounting_command.extend(["--input-file", str(input_file)])
    return [*accounting_command, "--", *(str(value) for value in command)]


def run_and_record(
    command,
    output_path,
    role,
    run_id,
    model_version=None,
    log_file=None,
    input_file=None,
    env=None,
):
    """Run a process and write a JSON resource record for its direct PID.

    On Linux, user CPU time, system CPU time, and peak resident memory are
    sampled from ``/proc/<pid>``. Other platforms still produce a record with
    wall-clock duration and clearly identify unavailable measurements.

    Parameters
    ----------
    command : list[str | pathlib.Path]
        Executable and arguments to run.
    output_path : str or pathlib.Path
        JSON resource-record output path.
    role : str
        Process role.
    run_id : int or str
        Production run identifier.
    model_version : str, optional
        Simulation-model version associated with this process.
    log_file : str or pathlib.Path, optional
        Gzip-compressed stdout log file. Stderr is written to a sibling
        ``.err.gz`` file, replacing the ``.log.gz`` suffix.
    input_file : str or pathlib.Path, optional
        File connected to the child's stdin.
    env : dict, optional
        Environment overrides for the child process.

    Returns
    -------
    int
        Child-process return code.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = datetime.now(UTC)
    start_clock = time.perf_counter()
    process, opened_files, log_writers = _start_process(command, log_file, input_file, env)
    capture_errors = []
    log_threads = _start_log_writers(process, log_writers, capture_errors)
    cpu_times = None
    peak_rss_bytes = None
    try:
        while process.poll() is None:
            cpu_times, peak_rss_bytes = _update_measurements(process.pid, cpu_times, peak_rss_bytes)
            time.sleep(0.05)
        cpu_times, peak_rss_bytes = _update_measurements(process.pid, cpu_times, peak_rss_bytes)
        return_code = process.wait()
    finally:
        for log_thread, process_output in log_threads:
            log_thread.join()
            process_output.close()
        for opened_file in opened_files:
            opened_file.close()

    record = _build_record(
        command,
        role,
        run_id,
        model_version,
        process.pid,
        start_time,
        datetime.now(UTC),
        time.perf_counter() - start_clock,
        cpu_times,
        peak_rss_bytes,
        return_code,
    )
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding=_ENCODING)
    if capture_errors:
        raise RuntimeError(
            "Failed to capture simulation output; inspect the available logs."
        ) from (capture_errors[0])
    return return_code


def _start_process(command, log_file, input_file, env):
    """Start the child process and return it with files needing closure."""
    opened_files = []
    log_writers = []
    stdin = None
    stdout = None
    stderr = None
    try:
        if input_file:
            stdin = Path(input_file).open("rb")  # pylint: disable=consider-using-with
            opened_files.append(stdin)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            error_path = log_path.with_name(
                log_path.name.removesuffix(_STDOUT_LOG_SUFFIX) + _STDERR_LOG_SUFFIX
            )
            for path in (log_path, error_path):
                log_writer = gzip.open(path, "wb")  # pylint: disable=consider-using-with
                opened_files.append(log_writer)
                log_writers.append(log_writer)
            stdout = subprocess.PIPE
            stderr = subprocess.PIPE
        # The process and streams must remain open until the monitoring loop completes.
        # pylint: disable=consider-using-with
        process = subprocess.Popen(
            [str(value) for value in command],
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            env=_build_environment(env),
        )
        # pylint: enable=consider-using-with
    except BaseException:
        for opened_file in opened_files:
            opened_file.close()
        raise
    return process, opened_files, log_writers


def _start_log_writers(process, log_writers, capture_errors):
    """Drain stdout and stderr concurrently into their separate gzip logs."""
    log_threads = []
    for process_output, log_writer in zip((process.stdout, process.stderr), log_writers):
        log_thread = threading.Thread(
            target=_copy_process_output,
            args=(process_output, log_writer, process, capture_errors),
        )
        log_thread.start()
        log_threads.append((log_thread, process_output))
    return log_threads


def _copy_process_output(process_output, log_writer, process, capture_errors):
    """Copy one stream, stopping the process if output cannot be retained."""
    try:
        while chunk := process_output.read(1024 * 1024):
            log_writer.write(chunk)
    except (OSError, ValueError, zlib.error) as exc:
        capture_errors.append(exc)
        try:
            process.kill()
        except ProcessLookupError:
            pass


def _build_environment(env):
    """Return the inherited process environment with optional overrides."""
    if env is None:
        return None
    process_env = os.environ.copy()
    process_env.update(env)
    return process_env


def _update_measurements(pid, cpu_times, peak_rss_bytes):
    """Read the latest direct-child CPU and resident-memory measurements."""
    sample = _read_linux_process_sample(pid)
    if sample is None:
        return cpu_times, peak_rss_bytes
    sampled_cpu_times, sampled_rss = sample
    if sampled_rss is not None:
        peak_rss_bytes = max(peak_rss_bytes or 0, sampled_rss)
    return sampled_cpu_times, peak_rss_bytes


def _read_linux_process_sample(pid):
    """Read CPU and resident-memory data for one PID from Linux procfs."""
    if platform.system() != "Linux":
        return None
    try:
        stat_fields = (
            Path(f"/proc/{pid}/stat").read_text(encoding=_ENCODING).rsplit(")", 1)[1].split()
        )
        status = Path(f"/proc/{pid}/status").read_text(encoding=_ENCODING)
    except FileNotFoundError, IndexError, PermissionError:
        return None
    clock_ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    cpu_times = {
        "user_seconds": int(stat_fields[11]) / clock_ticks,
        "system_seconds": int(stat_fields[12]) / clock_ticks,
    }
    return cpu_times, _read_peak_rss_bytes(status)


def _read_peak_rss_bytes(status):
    """Return the Linux peak resident set size in bytes from a status payload."""
    current_rss = None
    for line in status.splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
        if line.startswith("VmRSS:"):
            current_rss = int(line.split()[1]) * 1024
    return current_rss


def _build_record(
    command,
    role,
    run_id,
    model_version,
    pid,
    start_time,
    end_time,
    wall_time_seconds,
    cpu_times,
    peak_rss_bytes,
    return_code,
):
    """Build one JSON-serializable resource record."""
    return {
        "schema_name": "process_resource_record",
        "schema_version": _RECORD_SCHEMA_VERSION,
        "simtools_version": __version__,
        "run_id": str(run_id),
        "role": role,
        "model_version": model_version,
        "argv": [str(value) for value in command],
        "pid": pid,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "wall_time_seconds": wall_time_seconds,
        "user_cpu_seconds": None if cpu_times is None else cpu_times["user_seconds"],
        "system_cpu_seconds": None if cpu_times is None else cpu_times["system_seconds"],
        "peak_rss_bytes": peak_rss_bytes,
        "return_code": return_code,
        "signal": None if return_code >= 0 else signal.Signals(-return_code).name,
        "measurement_method": "linux_procfs_direct_pid"
        if platform.system() == "Linux"
        else "wall_clock_only",
        "platform": platform.platform(),
        **_execution_environment(),
    }


def _execution_environment():
    """Return host and container identity needed to interpret resource measurements."""
    container_image = next(
        (os.environ[name] for name in _CONTAINER_ENVIRONMENT_VARIABLES if os.environ.get(name)),
        None,
    )
    return {
        "hostname": platform.node() or None,
        "cpu_model": _read_cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "available_cpu_count": os.process_cpu_count(),
        "container_image": container_image,
    }


def _read_cpu_model():
    """Return the processor model name, or ``None`` when it is unavailable."""
    try:
        cpuinfo = _CPUINFO_PATH.read_text(encoding=_ENCODING)
    except OSError:
        return platform.processor() or None
    for line in cpuinfo.splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "model name" and value.strip():
            return value.strip()
    return platform.processor() or None


def _parse_command_line_arguments():
    """Parse process-accounting command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-version")
    parser.add_argument("--log-file")
    parser.add_argument("--input-file")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    if arguments.command[:1] == ["--"]:
        arguments.command = arguments.command[1:]
    if not arguments.command:
        parser.error("an executable command is required after '--'")
    return arguments


def main():
    """Run the requested command and return its exit status."""
    arguments = _parse_command_line_arguments()
    return run_and_record(
        arguments.command,
        arguments.output,
        arguments.role,
        arguments.run_id,
        model_version=arguments.model_version,
        log_file=arguments.log_file,
        input_file=arguments.input_file,
    )


if __name__ == "__main__":
    sys.exit(main())
