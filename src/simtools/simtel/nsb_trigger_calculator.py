"""Calculate NSB trigger rates from SIMTEL log files and generate ECSV output."""

import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from astropy.table import Table

from simtools.simtel.simtel_log_reader import (
    crawl_log_files,
    extract_event_count,
    extract_run_number,
    extract_trigger_count,
    read_log_file,
)

_logger = logging.getLogger(__name__)
_THRESHOLD_RE = re.compile(r"_[ad]sum(?P<threshold>\d+)(?=\.)")


def extract_threshold_from_file_name(file_path):
    """
    Extract threshold value from file name.

    Supports current production labels like:

    - ``*_asum220.simtel.log.gz``
    - ``*_dsum450.simtel.log.gz``

    Parameters
    ----------
    file_path : Path or str
        Path to the log file.

    Returns
    -------
    int or None
        Threshold value, or None if not found.
    """
    match = _THRESHOLD_RE.search(Path(file_path).name)
    if match:
        return int(match.group("threshold"))

    _logger.warning(f"Could not extract threshold from {file_path}")
    return None


def parse_nsb_log_file(file_path):
    """
    Parse a single NSB log file.

    The log file is read only for trigger and event counts.
    The threshold is extracted from the file name.

    Parameters
    ----------
    file_path : Path or str
        Path to the log file.

    Returns
    -------
    dict or None
        Dictionary with keys: 'run', 'threshold', 'triggers', 'events', 'file_path'.
        Returns None if critical information cannot be extracted.
    """
    file_path = Path(file_path)

    try:
        log_text = read_log_file(file_path)
    except Exception as e:  # pylint: disable=broad-exception-caught
        _logger.error(f"Failed to read {file_path}: {e}")
        return None

    run_number = extract_run_number(file_path)
    threshold = extract_threshold_from_file_name(file_path)
    triggers = extract_trigger_count(log_text)
    events = extract_event_count(log_text)

    if triggers is None and events is not None:
        _logger.info(f"No NSB triggers found in {file_path}; using 0 triggers")
        triggers = 0

    if run_number is None or threshold is None or triggers is None:
        _logger.warning(
            f"Skipping {file_path}: missing critical info "
            f"(run={run_number}, threshold={threshold}, triggers={triggers})"
        )
        return None

    return {
        "run": run_number,
        "threshold": threshold,
        "triggers": triggers,
        "events": events,
        "file_path": str(file_path),
    }


def parse_nsb_log_files(file_list):
    """
    Parse multiple NSB log files.

    Thresholds are extracted from file names, not directories.

    Parameters
    ----------
    file_list : list of Path
        List of log file paths to parse.

    Returns
    -------
    list of dict
        List of parsed data dictionaries. Failed parses are excluded.

    Raises
    ------
    ValueError
        If no log files could be parsed successfully.
    """
    data = []

    for file_path in file_list:
        _logger.debug(f"Parsing {file_path}")
        result = parse_nsb_log_file(file_path)
        if result is not None:
            data.append(result)

    _logger.info(f"Successfully parsed {len(data)} out of {len(file_list)} log files")

    if not data:
        raise ValueError("No log files could be parsed successfully")

    return data


def group_by_threshold_and_run(data):
    """
    Organize parsed data into a structure grouped by threshold and run.

    Parameters
    ----------
    data : list of dict
        List of parsed log file data.

    Returns
    -------
    dict
        Nested dictionary: {threshold: {run: {'triggers': int, 'events': int}}}
    """
    grouped = defaultdict(dict)

    for entry in data:
        threshold = entry["threshold"]
        run = entry["run"]
        grouped[threshold][run] = {
            "triggers": entry["triggers"],
            "events": entry["events"],
        }

    return dict(grouped)


def calculate_statistics(grouped_data, time_window):
    """
    Calculate statistics for each threshold.

    Parameters
    ----------
    grouped_data : dict
        Data grouped by threshold and run.
    time_window : float, optional
        Time window per event in seconds.

    Returns
    -------
    dict
        Statistics for each threshold with keys:
        - 'runs': dict of {run_number: triggers}
        - 'total_triggers': sum of all triggers
        - 'total_events': sum of all events
        - 'time_s': total time in seconds
        - 'rate_hz': trigger rate in Hz
        - 'rate_khz': trigger rate in kHz
        - 'error_hz': standard error of trigger counts in Hz
        - 'num_runs': number of runs
    """
    statistics = {}

    for threshold, runs_data in grouped_data.items():
        run_triggers = []
        run_events = []

        runs_dict = {}
        for run_num, run_info in sorted(runs_data.items()):
            triggers = run_info["triggers"]
            events = run_info["events"]

            runs_dict[run_num] = triggers
            run_triggers.append(triggers)

            if events is not None:
                run_events.append(events)

        total_triggers = np.sum(run_triggers)
        total_events = np.sum(run_events) if run_events else 0
        time_s = total_events * time_window if total_events > 0 else 0
        rate_hz = total_triggers / time_s if time_s > 0 else 0
        rate_khz = rate_hz / 1000.0
        error_hz = 0

        if len(run_triggers) > 1 and time_s > 0:
            std_dev = np.std(run_triggers, ddof=1)
            error_triggers = std_dev / np.sqrt(len(run_triggers))
            error_hz = error_triggers / time_s

        statistics[threshold] = {
            "runs": runs_dict,
            "total_triggers": int(total_triggers),
            "total_events": int(total_events),
            "time_s": time_s,
            "rate_hz": rate_hz,
            "rate_khz": rate_khz,
            "error_hz": error_hz,
            "num_runs": len(run_triggers),
        }

    return statistics


def generate_ecsv_output(statistics, output_file, time_window):
    """
    Generate ECSV table from statistics.

    Parameters
    ----------
    statistics : dict
        Statistics dictionary from calculate_statistics().
    output_file : Path or str
        Output file path for ECSV table.
    time_window : float, optional
        Time window per event in seconds for metadata.
    """
    output_file = Path(output_file)

    if not statistics:
        raise ValueError("No statistics to write")

    all_runs = set()
    for threshold_stats in statistics.values():
        all_runs.update(threshold_stats["runs"].keys())
    all_runs = sorted(all_runs)

    threshold_col = []
    run_cols = {run: [] for run in all_runs}
    total_triggers_col = []
    total_events_col = []
    time_col = []
    rate_hz_col = []
    rate_khz_col = []
    error_hz_col = []

    for threshold in sorted(statistics.keys()):
        stats = statistics[threshold]

        threshold_col.append(threshold)

        for run in all_runs:
            if run in stats["runs"]:
                run_cols[run].append(stats["runs"][run])
            else:
                run_cols[run].append(np.nan)

        total_triggers_col.append(stats["total_triggers"])
        total_events_col.append(stats["total_events"])
        time_col.append(stats["time_s"])
        rate_hz_col.append(stats["rate_hz"])
        rate_khz_col.append(stats["rate_khz"])
        error_hz_col.append(stats["error_hz"])

    table_data = {"threshold": threshold_col}

    for run in all_runs:
        table_data[f"run{run}"] = run_cols[run]

    table_data["Total trig'd"] = total_triggers_col
    table_data["Out of"] = total_events_col
    table_data["Time (s)"] = time_col
    table_data["Rate (Hz)"] = rate_hz_col
    table_data["Rate (kHz)"] = rate_khz_col
    table_data["Error (Hz)"] = error_hz_col

    table = Table(table_data)

    total_events_sum = sum(total_events_col)
    table.meta["comments"] = [
        f"Total events: {total_events_sum}",
        f"Time window per event: {time_window} s",
    ]

    for col in ["Time (s)", "Rate (Hz)", "Rate (kHz)", "Error (Hz)"]:
        table[col].format = ".2f"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    table.write(output_file, format="ascii.ecsv", overwrite=True)

    _logger.info(f"ECSV table written to {output_file}")
    _logger.info(f"Table contains {len(threshold_col)} thresholds and {len(all_runs)} runs")


def derive_nsb_triggers(args):
    """
    Derive NSB trigger rates from log files.

    Parameters
    ----------
    args : dict
        Configuration parameters with keys:
        - root_dir: Root directory to search for log files
        - pattern: Glob pattern for log files, default: ``**/*.simtel.log.gz``
        - output: Output ECSV file path (optional, if None, no file is written)
        - time_window: Time window per event in seconds (default: 66.4e-9)
        - verbose: Enable verbose logging (optional)

    Returns
    -------
    dict
        Statistics dictionary with threshold as keys, containing:
        - rate_hz: trigger rate in Hz
        - rate_khz: trigger rate in kHz
        - error_hz: standard error in Hz
        - total_triggers: total number of triggers
        - num_runs: number of runs

    Raises
    ------
    FileNotFoundError
        If root directory doesn't exist or no log files found.
    ValueError
        If no log files could be parsed successfully.
    """
    pattern = args.get("pattern", "**/*.simtel.log.gz")
    time_window = args.get("time_window")

    _logger.info("NSB Trigger Rate Calculator")
    _logger.info(f"Root directory: {args['root_dir']}")
    _logger.info(f"Time window: {time_window} s")
    if args.get("output"):
        _logger.info(f"Output file: {args['output']}")

    _logger.info("Searching for log files...")
    log_files = crawl_log_files(args["root_dir"], pattern)
    _logger.info(f"Found {len(log_files)} log files")

    _logger.info("Parsing log files...")
    parsed_data = parse_nsb_log_files(log_files)
    _logger.info(f"Successfully parsed {len(parsed_data)} log files")

    _logger.info("Grouping data by threshold and run...")
    grouped_data = group_by_threshold_and_run(parsed_data)
    thresholds = sorted(grouped_data.keys())
    _logger.info(f"Found {len(thresholds)} thresholds: {thresholds}")

    _logger.info("Calculating statistics...")
    statistics = calculate_statistics(grouped_data, time_window)

    _logger.info("Summary:")
    for threshold in sorted(statistics.keys()):
        stats = statistics[threshold]
        _logger.info(
            f"  Threshold {threshold}: "
            f"{stats['num_runs']} runs, "
            f"{stats['total_triggers']} triggers, "
            f"{stats['rate_hz']:.2f} Hz"
        )

    if args.get("output"):
        _logger.info("Writing ECSV output...")
        generate_ecsv_output(statistics, args["output"], time_window)
        _logger.info(f"Output written to: {args['output']}")

    _logger.info("NSB trigger rate calculation completed")

    return statistics
