"""Generic helpers for matching and validating supported file types."""

from pathlib import Path

FILE_TYPE_SUFFIXES = {
    "hdf5": (".hdf5", ".h5"),
    "json_or_yaml": (".json", ".yml", ".yaml"),
    "table": (".ecsv", ".fits", ".fits.gz"),
    "sim_telarray": (".simtel", ".simtel.gz", ".simtel.zst"),
}


def validate_file_type(file_path, file_type):
    """
    Validate that a file has one of the expected suffixes.

    Parameters
    ----------
    file_path : str or Path
        Path to the file to validate.
    file_type : str
        Registered file type name, e.g., "hdf5", "json_or_yaml",
    """
    path = Path(file_path)
    expected_suffixes = _suffixes_for_file_type(file_type)
    if not matches_suffix(path, expected_suffixes):
        raise ValueError(
            f"File '{file_path}' has unsupported suffix, expected one of {expected_suffixes}"
        )
    return path


def _suffixes_for_file_type(file_type):
    """Return registered suffixes for a named file type."""
    try:
        return FILE_TYPE_SUFFIXES[file_type]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported file type '{file_type}', expected one of {sorted(FILE_TYPE_SUFFIXES)}"
        ) from exc


def matches_suffix(file_path, expected_suffixes):
    """Return whether the path ends with one of the expected suffix patterns."""
    name = Path(file_path).name.lower()
    return any(name.endswith(suffix) for suffix in expected_suffixes)


def is_file_type(file_path, file_type):
    """Return whether the path matches one registered file type."""
    return matches_suffix(file_path, _suffixes_for_file_type(file_type))


def looks_like_text_file(file_path, sample_size=4096):
    """Return whether the file appears to be UTF-8 text."""
    try:
        sample = Path(file_path).read_bytes()[:sample_size]
    except OSError:
        return False
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True
