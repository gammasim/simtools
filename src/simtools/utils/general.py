"""General functions useful across different parts of the code."""

import datetime as dt
import glob
import logging
import os
import uuid
from pathlib import Path
from urllib.parse import urlparse

import dotenv

_logger = logging.getLogger(__name__)


def cleanup_intermediate_files(output_dir, patterns=(), *, files=(), directories=(), exclude=()):
    """Remove selected intermediate files and then selected empty directories.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Directory in which to match glob patterns.
    patterns : iterable of str, optional
        File patterns relative to output_dir. No patterns are selected by default.
    files : iterable of pathlib.Path, optional
        Explicit file paths to remove, independent of output_dir.
    directories : iterable of pathlib.Path, optional
        Explicit directories to remove only when empty, deepest first.
    exclude : iterable of pathlib.Path, optional
        Paths to preserve even when selected by a pattern or explicit list.

    Returns
    -------
    int
        Number of files removed. Missing files are ignored.
    """
    selected = {Path(path) for path in files}
    for pattern in patterns:
        selected.update(Path(output_dir).glob(pattern))
    selected.difference_update(Path(path) for path in exclude)
    empty_directories = {Path(path) for path in directories} | {
        path for path in selected if path.is_dir() and not path.is_symlink()
    }
    empty_directories.difference_update(Path(path) for path in exclude)
    removed = 0
    for path in selected:
        if path.is_file() or path.is_symlink():
            path.unlink(missing_ok=True)
            removed += 1
            _logger.debug("Removed: %s", path)
    for directory in sorted(empty_directories, key=lambda path: len(path.parts), reverse=True):
        if directory.is_dir() and not directory.is_symlink() and not any(directory.iterdir()):
            directory.rmdir()
    if removed:
        _logger.info("Cleanup: removed %s intermediate files", removed)
    return removed


def is_url(url):
    """
    Check if a string is a valid URL.

    Parameters
    ----------
    url: str
        String to be checked.

    Returns
    -------
    bool
        True if url is a valid URL.

    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except AttributeError:
        return False


def _search_directory(directory, filename, rec=False):
    if not Path(directory).exists():
        _logger.debug(f"Directory {directory} does not exist")
        return None

    _file = Path(directory).joinpath(filename)
    if _file.exists():
        _logger.debug(f"File {filename} found in {directory}")
        return _file

    if rec:
        for subdir in Path(directory).iterdir():
            if subdir.is_dir():
                _file = _search_directory(subdir, filename, True)
                if _file:
                    return _file
    return None


def find_executable_in_dir(name, directory):
    """
    Find an executable in the given directory.

    Parameters
    ----------
    name: str
        Name of the executable to find.
    directory: str or Path
        Directory to search for the executable.
    """
    if name is None or directory is None:
        raise ValueError("Both name and directory must be provided.")
    path = Path(directory) / name
    if not path.is_file():
        raise FileNotFoundError(f"Executable not found: {path}")
    if not os.access(path, os.X_OK):
        raise PermissionError(f"Not executable: {path}")
    return path


def find_file(name, loc):
    """
    Search for files inside of given directories, recursively, and return its full path.

    Parameters
    ----------
    name: str
        File name to be searched for.
    loc: Path or list of Path
        Location of where to search for the file.

    Returns
    -------
    Path
        Full path of the file to be found if existing. Otherwise, None.

    Raises
    ------
    FileNotFoundError
        If the desired file is not found.
    """
    all_locations = [loc] if not isinstance(loc, list) else loc

    # Searching file locally
    _file = _search_directory(".", name)
    if _file:
        return _file

    # Searching file in given locations
    for location in all_locations:
        _file = _search_directory(location, name, True)
        if _file:
            return _file

    msg = f"File {name} could not be found in {all_locations}"
    _logger.error(msg)
    raise FileNotFoundError(msg)


def resolve_file_patterns(file_names):
    """
    Return a list of files names from string, list, or wildcard pattern.

    Parameters
    ----------
    file_names: str, list
        File names to be searched for (wildcards allowed).

    Returns
    -------
    list
        List of file names found.
    """
    if file_names is None:
        raise ValueError("No file list provided.")
    if not isinstance(file_names, list):
        file_names = [file_names]

    _files = []
    for file_name in file_names:
        # use glob (and not Path.glob) for easier wildcard handling
        _files.extend(Path(f) for f in glob.glob(str(file_name), recursive=True))  # noqa: PTH207
    if not _files:
        raise FileNotFoundError(f"No files found: {file_names}")
    return _files


def now_date_time_in_isoformat():
    """Return date and time in isoformat and second accuracy."""
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def load_environment_variables(env_file=".env", env_list=None):
    """
    Load environment variables (from a .env file or directly from the environment).

    Allow to read a specific list of variables or all variables from the .env file.

    Parameters
    ----------
    env_file: str
        Path to the .env file.
    env_list: list, optional
        List of environment variables to be read. If None, all variables are read.

    Returns
    -------
    dict
        Dictionary mapping environment variable names (lowercase, without the
        ``SIMTOOLS_`` prefix) to their cleaned string values.
    """
    dotenv.load_dotenv(env_file or None)
    keys = (
        list(dotenv.dotenv_values(env_file).keys())
        if env_list is None
        else [f"SIMTOOLS_{s.upper()}" for s in env_list]
    )

    env_values = {}
    for key in keys:
        env_value = os.environ.get(key)
        if env_value is None:
            continue
        cleaned_value = env_value.split("#")[0].strip().replace('"', "").replace("'", "")
        env_values[key.removeprefix("SIMTOOLS_").lower()] = cleaned_value
    return env_values


def get_uuid():
    """
    Generate a UUID (7) string.

    Returns
    -------
    str
        A UUID string.
    """
    return str(uuid.uuid7())


def extract_uuid7_from_path(path):
    """Extract UUID7 from path components if present.

    Parameters
    ----------
    path : str or Path
        Path potentially containing a UUID7 component.

    Returns
    -------
    str or None
        UUID7 string if found, otherwise None.
    """
    for path_part in reversed(Path(path).parts):
        try:
            candidate = uuid.UUID(path_part)
        except ValueError, TypeError:
            continue
        if candidate.version == 7:
            return str(candidate)
    return None


def extract_subdirectories_from_path(path, anchor="input"):
    """Extract subdirectories in a path after an anchor directory.

    Parameters
    ----------
    path : str or Path
        Path containing an anchor directory and a filename.
    anchor : str
        Directory name used as extraction anchor.

    Returns
    -------
    str
        Subdirectory path between anchor and file name, joined with '/'.

    Raises
    ------
    ValueError
        If anchor is not present or no subdirectories are found after the anchor.
    """
    path = Path(path)
    try:
        anchor_index = path.parts.index(anchor)
        subdirs = path.parts[anchor_index + 1 : -1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Could not find subdirectory under '{anchor}': {exc}") from exc

    if len(subdirs) == 0:
        raise ValueError(f"Could not find subdirectory under '{anchor}'")

    return str(Path(*subdirs))
