"""Generic helpers for matching and validating supported file types."""

from pathlib import Path

import simtools.utils.general as gen

FILE_TYPE_SUFFIXES = {
    "hdf5": (".hdf5", ".h5"),
    "json_or_yaml": (".json", ".yml", ".yaml"),
    "table": (".ecsv", ".fits", ".fits.gz"),
    "sim_telarray": (".simtel", ".simtel.gz", ".simtel.zst"),
}


def validate_file_type(file_path, expected_suffixes):
    """Validate that a file has one of the expected suffixes."""
    path = Path(file_path)
    expected_suffixes = gen.ensure_list(expected_suffixes)
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


def is_path_type(file_path, file_type):
    """Return whether the path matches one registered file type."""
    return matches_suffix(file_path, _suffixes_for_file_type(file_type))


def validate_path_type(file_path, file_type):
    """Validate one file path against a registered file type."""
    return validate_file_type(file_path, _suffixes_for_file_type(file_type))


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
