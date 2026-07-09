"""Inspect simtools-related files and return structured summaries."""

from importlib import import_module
from pathlib import Path

import h5py
from astropy.table import Table

from simtools.io import ascii_handler
from simtools.io.file_type import is_path_type, looks_like_text_file, validate_path_type
from simtools.simtel import simtel_io_metadata


def inspect_file(file_path, max_entries=50, format_report=True):
    """
    Inspect one supported file and return one or more reports.

    Parameters
    ----------
    file_path : str or Path
        Path to the file to inspect.
    max_entries : int, optional
        Maximum number of entries to include in collection-style summaries.
    format_report : bool, optional
        Return formatted strings instead of raw dictionaries.

    Returns
    -------
    list[str] or list[dict]
        Inspection reports. HDF5 trigger-histogram files may yield both a generic
        HDF5 report and a trigger-histogram-specific consistency report.
    """
    file_path = Path(file_path)
    max_entries = _normalize_max_entries(max_entries)
    inspector = _select_inspector(file_path)
    reports = [inspector(file_path, max_entries=max_entries, format_report=format_report)]

    if is_path_type(file_path, "hdf5"):
        trigger_histogram_report = _inspect_trigger_histogram_file(
            file_path,
            format_report=format_report,
        )
        if trigger_histogram_report is not None:
            reports.append(trigger_histogram_report)
    return reports


def inspect_hdf5_file(file_path, max_entries=50, format_report=True):
    """Inspect one HDF5 file and return a report."""
    file_path = validate_path_type(file_path, "hdf5")
    if not h5py.is_hdf5(file_path):
        raise ValueError(f"File '{file_path}' is not a valid HDF5 container.")

    with h5py.File(file_path, "r") as hdf5_file:
        entries = _collect_hdf5_entries(hdf5_file)
        root_entries = sorted(hdf5_file.keys())

    report = {
        "file_path": Path(file_path),
        "file_type": "hdf5",
        "root_entries": root_entries,
        "group_count": sum(1 for entry in entries if entry["kind"] == "group"),
        "dataset_count": sum(1 for entry in entries if entry["kind"] == "dataset"),
        "entries": entries[:max_entries],
        "entries_truncated": _is_truncated(entries, max_entries),
        "total_entries": len(entries),
    }
    return _format_hdf5_report(report) if format_report else report


def inspect_json_or_yaml_file(file_path, max_entries=50, format_report=True):
    """Inspect one JSON or YAML file and return a report."""
    del max_entries
    suffix = Path(file_path).suffix.lower()
    file_path = validate_path_type(file_path, "json_or_yaml")
    data = ascii_handler.collect_data_from_file(file_path)
    report = {
        "file_path": Path(file_path),
        "file_type": "json" if suffix == ".json" else "yaml",
        "top_level_type": type(data).__name__,
        "top_level_keys": list(data.keys()) if isinstance(data, dict) else None,
        "item_count": len(data) if hasattr(data, "__len__") else None,
    }
    return _format_key_value_report(report) if format_report else report


def inspect_table_file(file_path, max_entries=50, format_report=True):
    """Inspect one tabular file and return a report."""
    file_path = validate_path_type(file_path, "table")
    table = Table.read(file_path)
    report = {
        "file_path": Path(file_path),
        "file_type": "fits" if str(file_path).lower().endswith((".fits", ".fits.gz")) else "ecsv",
        "row_count": len(table),
        "column_count": len(table.colnames),
        "columns": table.colnames[:max_entries],
        "columns_truncated": _is_truncated(table.colnames, max_entries),
    }
    return _format_table_report(report) if format_report else report


def inspect_sim_telarray_file(file_path, max_entries=50, format_report=True):
    """Inspect one sim_telarray file and return a report."""
    file_path = validate_path_type(file_path, "sim_telarray")
    if not is_path_type(file_path, "sim_telarray"):
        raise ValueError(f"File '{file_path}' has unsupported suffix for sim_telarray inspection.")
    global_meta, telescope_meta = simtel_io_metadata.read_sim_telarray_metadata(file_path)
    sorted_global_items = sorted(global_meta.items()) if global_meta is not None else []
    telescope_ids = sorted(telescope_meta.keys())
    preview_telescope_ids = telescope_ids[:max_entries]
    report = {
        "file_path": file_path,
        "file_type": "sim_telarray",
        "file_size_bytes": file_path.stat().st_size,
        "global_metadata": dict(sorted_global_items[:max_entries]),
        "global_metadata_truncated": _is_truncated(sorted_global_items, max_entries),
        "telescope_count": len(telescope_ids),
        "telescope_ids": preview_telescope_ids,
        "telescope_ids_truncated": _is_truncated(telescope_ids, max_entries),
    }
    return _format_simtel_report(report) if format_report else report


def inspect_text_file(file_path, max_entries=50, format_report=True):
    """Inspect one plain-text file and return a report."""
    file_path = Path(file_path)
    text = file_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    report = {
        "file_path": file_path,
        "file_type": "text",
        "line_count": len(lines),
        "preview_lines": lines[:max_entries],
        "preview_truncated": _is_truncated(lines, max_entries),
    }
    return _format_text_report(report) if format_report else report


def _select_inspector(file_path):
    """Return the inspector function appropriate for the file path."""
    if is_path_type(file_path, "hdf5"):
        return inspect_hdf5_file
    if is_path_type(file_path, "json_or_yaml"):
        return inspect_json_or_yaml_file
    if is_path_type(file_path, "table"):
        return inspect_table_file
    if is_path_type(file_path, "sim_telarray"):
        return inspect_sim_telarray_file
    if looks_like_text_file(file_path):
        return inspect_text_file
    raise ValueError(f"Unsupported file type for inspection: {file_path.suffix or '<no suffix>'}.")


def _inspect_trigger_histogram_file(file_path, format_report=True):
    """Import and run trigger-histogram inspection lazily to avoid import cycles."""
    module = import_module("simtools.production_configuration.trigger_histograms")
    return module.inspect_trigger_histogram_file(file_path, format_report=format_report)


def _normalize_max_entries(max_entries):
    """Normalize max_entries so non-positive values disable truncation."""
    if max_entries is None or max_entries <= 0:
        return None
    return max_entries


def _is_truncated(values, max_entries):
    """Return whether a collection exceeds the configured preview limit."""
    return max_entries is not None and len(values) > max_entries


def _format_hdf5_report(report):
    """Format an HDF5 inspection report."""
    lines = [
        f"File: {report['file_path']}",
        f"Detected file type: {report['file_type']}",
        f"Root entries ({len(report['root_entries'])}):",
    ]
    lines.extend(f"- {entry}" for entry in report["root_entries"])
    lines.append(
        f"Contained objects: {report['group_count']} groups, {report['dataset_count']} datasets"
    )
    lines.append(f"Listing first {len(report['entries'])} of {report['total_entries']} entries:")
    for entry in report["entries"]:
        line = f"- {entry['path']} [{entry['kind']}]"
        if entry["kind"] == "dataset":
            line += f" shape={entry['shape']} dtype={entry['dtype']}"
        lines.append(line)
    if report["entries_truncated"]:
        lines.append("- ... output truncated ...")
    return "\n".join(lines)


def _format_key_value_report(report):
    """Format a generic key-value inspection report."""
    lines = [
        f"File: {report['file_path']}",
        f"Detected file type: {report['file_type']}",
    ]
    for key, value in report.items():
        if key in {"file_path", "file_type"}:
            continue
        if value is None:
            continue
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _format_table_report(report):
    """Format a tabular-file inspection report."""
    lines = [
        f"File: {report['file_path']}",
        f"Detected file type: {report['file_type']}",
        f"Rows: {report['row_count']}",
        f"Columns ({report['column_count']}): {', '.join(report['columns'])}",
    ]
    if report["columns_truncated"]:
        lines.append("Column list truncated.")
    return "\n".join(lines)


def _format_text_report(report):
    """Format a plain-text inspection report."""
    lines = [
        f"File: {report['file_path']}",
        f"Detected file type: {report['file_type']}",
        f"Lines: {report['line_count']}",
        f"Preview ({len(report['preview_lines'])} lines):",
    ]
    lines.extend(f"- {line}" for line in report["preview_lines"])
    if report["preview_truncated"]:
        lines.append("- ... output truncated ...")
    return "\n".join(lines)


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


def _collect_hdf5_entries(hdf5_file):
    """Collect all groups and datasets from one HDF5 file."""
    entries = []

    def _visitor(name, obj):
        path = f"/{name}"
        if isinstance(obj, h5py.Group):
            entries.append({"path": path, "kind": "group"})
        elif isinstance(obj, h5py.Dataset):
            entries.append(
                {
                    "path": path,
                    "kind": "dataset",
                    "shape": tuple(obj.shape),
                    "dtype": str(obj.dtype),
                }
            )

    hdf5_file.visititems(_visitor)
    return entries
