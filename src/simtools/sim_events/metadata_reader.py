"""Read CORSIKA event-file headers as format-neutral input-metadata fields."""

from collections.abc import Mapping

from simtools.io import ascii_handler
from simtools.sim_events.file_info import get_corsika_run_and_event_headers


def read_metadata(metadata_file_name):
    """Return input-metadata fields of a CORSIKA file (see ``MetadataCollector``)."""
    run_header, event_header = get_corsika_run_and_event_headers(metadata_file_name)
    if run_header is None or event_header is None:
        raise ValueError(f"CORSIKA file has no complete run and event header: {metadata_file_name}")
    run_header = _header_to_dict(run_header)
    event_header = _header_to_dict(event_header)
    version = run_header.get("version")
    return {
        "software_name": "corsika",
        "software_version": (f"{float(version):.4f}" if version is not None else None),
        "format_name": "corsika",
        "raw_metadata": {"run_header": run_header, "event_header": event_header},
    }


def _header_to_dict(header):
    """Convert a parsed CORSIKA header to a JSON-compatible dictionary."""
    if header is None:
        return {}
    if isinstance(header, Mapping):
        return ascii_handler.to_builtin(header)

    field_names = getattr(getattr(header, "dtype", None), "names", None)
    if field_names:
        record = header[0] if getattr(header, "shape", ()) else header
        return {name: ascii_handler.to_builtin(record[name]) for name in field_names}
    return ascii_handler.to_builtin(header)
