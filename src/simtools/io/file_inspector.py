"""Inspect simulation-related files, starting with HDF5 containers."""

from pathlib import Path

import h5py

import simtools.utils.general as gen


def inspect_file(file_path, max_entries=50):
    """Inspect one supported file and return a structured report."""
    file_path = gen.validate_file_type(file_path, [".hdf5", ".h5"])
    if not h5py.is_hdf5(file_path):
        raise ValueError(f"File '{file_path}' is not a valid HDF5 container.")
    return inspect_hdf5_file(file_path, max_entries=max_entries)


def inspect_hdf5_file(file_path, max_entries=50):
    """Inspect one HDF5 file and return a structured report."""
    file_path = Path(file_path)
    with h5py.File(file_path, "r") as hdf5_file:
        entries = _collect_hdf5_entries(hdf5_file)
        root_entries = sorted(hdf5_file.keys())

    return {
        "file_path": file_path,
        "file_type": "hdf5",
        "root_entries": root_entries,
        "group_count": sum(1 for entry in entries if entry["kind"] == "group"),
        "dataset_count": sum(1 for entry in entries if entry["kind"] == "dataset"),
        "entries": entries[:max_entries],
        "entries_truncated": len(entries) > max_entries,
        "total_entries": len(entries),
    }


def format_inspection_report(report):
    """Format a structured report as human-readable text."""
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
