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
from datetime import UTC, datetime
from pathlib import Path


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
        Gzip-compressed combined stdout and stderr log file.
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
        Gzip-compressed combined stdout and stderr log file.
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
    process, opened_files, log_writer = _start_process(command, log_file, input_file, env)
    log_thread = _start_log_writer(process, log_writer)
    cpu_times = None
    peak_rss_bytes = None
    try:
        while process.poll() is None:
            cpu_times, peak_rss_bytes = _update_measurements(process.pid, cpu_times, peak_rss_bytes)
            time.sleep(0.05)
        cpu_times, peak_rss_bytes = _update_measurements(process.pid, cpu_times, peak_rss_bytes)
        return_code = process.wait()
    finally:
        if log_thread:
            log_thread.join()
            process.stdout.close()
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
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return return_code


def _start_process(command, log_file, input_file, env):
    """Start the child process and return it with files needing closure."""
    opened_files = []
    stdin = None
    stdout = None
    stderr = None
    if input_file:
        stdin = Path(input_file).open("rb")  # pylint: disable=consider-using-with
        opened_files.append(stdin)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_writer = gzip.open(log_path, "wb")  # pylint: disable=consider-using-with
        opened_files.append(log_writer)
        stdout = subprocess.PIPE
        stderr = subprocess.STDOUT
    else:
        log_writer = None
    try:
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
    return process, opened_files, log_writer


def _start_log_writer(process, log_writer):
    """Copy combined child output to a gzip log without buffering it in memory."""
    if log_writer is None:
        return None
    log_thread = threading.Thread(target=_copy_process_output, args=(process.stdout, log_writer))
    log_thread.start()
    return log_thread


def _copy_process_output(process_output, log_writer):
    """Copy one child-process output stream to its compressed log file."""
    while chunk := process_output.read(1024 * 1024):
        log_writer.write(chunk)


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
            Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        )
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
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
        "schema_version": "1.0.0",
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
    }


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
