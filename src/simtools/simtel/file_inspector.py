"""Inspect sim_telarray output files."""

from pathlib import Path

from simtools.io.file_type import is_file_type, validate_file_type
from simtools.simtel import simtel_io_metadata


def inspect_sim_telarray_file(file_path, max_entries=50, format_report=True):
    """Inspect one sim_telarray file and return a report."""
    file_path = validate_file_type(file_path, "sim_telarray")
    if not is_file_type(file_path, "sim_telarray"):
        raise ValueError(f"File '{file_path}' has unsupported suffix for sim_telarray inspection.")
    global_meta, telescope_meta = simtel_io_metadata.read_sim_telarray_metadata(file_path)
    sorted_global_items = sorted(global_meta.items()) if global_meta is not None else []
    telescope_ids = sorted(telescope_meta.keys())
    report = {
        "file_path": Path(file_path),
        "file_type": "sim_telarray",
        "file_size_bytes": file_path.stat().st_size,
        "global_metadata": dict(sorted_global_items[:max_entries]),
        "global_metadata_truncated": _is_truncated(sorted_global_items, max_entries),
        "telescope_count": len(telescope_ids),
        "telescope_ids": telescope_ids[:max_entries],
        "telescope_ids_truncated": _is_truncated(telescope_ids, max_entries),
    }
    return _format_simtel_report(report) if format_report else report


def _format_simtel_report(report):
    """Format a sim_telarray inspection report."""
    lines = [
        f"File: {report['file_path']}",
        f"Detected file type: {report['file_type']}",
        f"file_size_bytes: {report['file_size_bytes']}",
        f"telescope_count: {report['telescope_count']}",
        f"telescope_ids: {report['telescope_ids']}",
        "global_metadata:",
    ]
    if report["global_metadata"]:
        lines.extend(f"- {key}: {value}" for key, value in report["global_metadata"].items())
    else:
        lines.append("- none")
    if report["global_metadata_truncated"]:
        lines.append("- ... metadata truncated ...")
    if report["telescope_ids_truncated"]:
        lines.append("telescope_ids_truncated: True")
    return "\n".join(lines)


def _is_truncated(values, max_entries):
    """Return whether a collection exceeds the configured preview limit."""
    return max_entries is not None and 0 < max_entries < len(values)
