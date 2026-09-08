"""Naming helpers for exported simulation-model assets."""

from pathlib import Path

ECSV_SUFFIX = ".ecsv"
SOURCE_VALUE_KEY = "_simtools_export_source_value"


def get_export_file_name(parameter_data, fallback_instrument=None):
    """Return the deterministic exported name for a model asset.

    ECSV assets are qualified with the instrument scope so that assets from
    different telescope designs can share one model directory safely. The
    original source value is kept separately when a parameter has already
    been qualified.

    Parameters
    ----------
    parameter_data : dict
        Model parameter metadata.
    fallback_instrument : str, optional
        Instrument scope to use when the metadata does not contain one.

    Returns
    -------
    str
        Exported filename.
    """
    value = parameter_data.get(SOURCE_VALUE_KEY, parameter_data.get("value"))
    if not isinstance(value, str) or not value.lower().endswith(ECSV_SUFFIX):
        return value

    instrument = parameter_data.get("instrument") or fallback_instrument or "global"
    path = Path(value)
    return f"{path.stem}-{instrument}{path.suffix}"


def qualify_parameter_file_name(parameter_data, fallback_instrument=None):
    """Update ECSV parameter metadata with its deterministic exported name."""
    value = parameter_data.get("value")
    qualified_value = get_export_file_name(parameter_data, fallback_instrument)
    if qualified_value != value and isinstance(value, str):
        parameter_data.setdefault(SOURCE_VALUE_KEY, value)
        parameter_data["value"] = qualified_value
    return qualified_value
