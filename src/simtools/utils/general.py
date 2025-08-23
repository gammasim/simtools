"""General functions useful across different parts of the code."""

import datetime
import glob
import logging
import os
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

__all__ = [
    "change_dict_keys_case",
    "clear_default_sim_telarray_cfg_directories",
    "collect_final_lines",
    "collect_kwargs",
    "get_log_excerpt",
    "get_log_level_from_user",
    "remove_substring_recursively_from_dict",
    "set_default_kwargs",
]

_logger = logging.getLogger(__name__)


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


def url_exists(url):
    """
    Check if a URL exists.

    Parameters
    ----------
    url: str
        URL to be checked.

    Returns
    -------
    bool
        True if URL exists.
    """
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status == 200
    except (urllib.error.URLError, AttributeError) as e:
        _logger.error(f"URL {url} does not exist: {e}")
        return False


def collect_kwargs(label, in_kwargs):
    """
    Collect kwargs of the type label_* and return them as a dict.

    Parameters
    ----------
    label: str
        Label to be collected in kwargs.
    in_kwargs: dict
        kwargs.

    Returns
    -------
    dict
        Dictionary with the collected kwargs.
    """
    out_kwargs = {}
    for key, value in in_kwargs.items():
        if label + "_" in key:
            out_kwargs[key.replace(label + "_", "")] = value
    return out_kwargs


def set_default_kwargs(in_kwargs, **kwargs):
    """
    Fill in a dict with a set of default kwargs and return it.

    Parameters
    ----------
    in_kwargs: dict
        Input dict to be filled in with the default kwargs.
    **kwargs:
        Default kwargs to be set.

    Returns
    -------
    dict
        Dictionary containing the default kwargs.
    """
    for par, value in kwargs.items():
        if par not in in_kwargs.keys():
            in_kwargs[par] = value
    return in_kwargs


def collect_final_lines(file, n_lines):
    """
    Collect final lines.

    Parameters
    ----------
    file: str or Path
        File to collect the lines from.
    n_lines: int
        Number of lines to be collected.

    Returns
    -------
    str
        Final lines collected.
    """
    list_of_lines = []

    if Path(file).suffix == ".gz":
        import gzip  # pylint: disable=import-outside-toplevel

        file_open_function = gzip.open
    else:
        file_open_function = open
    with file_open_function(file, "rb") as read_obj:
        # Move the cursor to the end of the file
        read_obj.seek(0, os.SEEK_END)
        # Create a buffer to keep the last read line
        buffer = bytearray()
        # Get the current position of pointer i.e eof
        pointer_location = read_obj.tell()
        # Loop till pointer reaches the top of the file
        while pointer_location >= 0:
            # Move the file pointer to the location pointed by pointer_location
            read_obj.seek(pointer_location)
            # Shift pointer location by -1
            pointer_location = pointer_location - 1
            # read that byte / character
            new_byte = read_obj.read(1)
            # If the read byte is new line character then it means one line is read
            if new_byte == b"\n":
                # Save the line in list of lines
                list_of_lines.append(buffer.decode()[::-1])
                # If the size of list reaches n_lines, then return the reversed list
                if len(list_of_lines) == n_lines:
                    return "".join(list(reversed(list_of_lines)))
                # Reinitialize the byte array to save next line
                buffer = bytearray()
            else:
                # If last read character is not eol then add it in buffer
                buffer.extend(new_byte)
        # As file is read completely, if there is still data in buffer, then its first line.
        if len(buffer) > 0:
            list_of_lines.append(buffer.decode()[::-1])

    return "".join(list(reversed(list_of_lines)))


def get_log_level_from_user(log_level):
    """
    Map between logging level from the user to logging levels of the logging module.

    Parameters
    ----------
    log_level: str
        Log level from the user.

    Returns
    -------
    logging.LEVEL
        The requested logging level to be used as input to logging.setLevel().
    """
    possible_levels = {
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    try:
        log_level_lower = log_level.lower()
    except AttributeError:
        log_level_lower = log_level
    if log_level_lower not in possible_levels:
        raise ValueError(
            f"'{log_level}' is not a logging level, "
            f"only possible ones are {list(possible_levels.keys())}"
        )

    return possible_levels[log_level_lower]


def ensure_iterable(value):
    """
    Return input value as iterable.

    - Single values will return as a list with a single element.
    - None values will return as empty list.
    - Values of list or tuple type are not changed.

    Parameters
    ----------
    value: any
        Input value to be converted to a iterable.

    Returns
    -------
    list or tuple
        Converted value as list or tuple.
    """
    if not value:
        return []
    return value if isinstance(value, list | tuple) else [value]


def program_is_executable(program):
    """
    Check if program exists and is executable.

    Follows https://stackoverflow.com/questions/377017/

    """
    program = Path(program)

    def is_exe(fpath):
        return fpath.is_file() and os.access(fpath, os.X_OK)

    fpath, _ = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        try:
            for path in os.environ["PATH"].split(os.pathsep):
                exe_file = Path(path) / program
                if is_exe(exe_file):
                    return exe_file
        except KeyError:
            _logger.warning("PATH environment variable is not set.")
            return None

    return None


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


def pack_tar_file(tar_file_name, file_list):
    """
    Pack files into a tar.gz archive.

    Parameters
    ----------
    tar_file_name: str
        Name of the output tar.gz file.
    file_list: list
        List of files to include in the archive.
    """
    file_list = [Path(f) for f in file_list]
    base = Path(os.path.commonpath([f.resolve() for f in file_list]))
    for f in file_list:
        if not f.is_file() or not f.resolve().is_relative_to(base.resolve()):
            raise ValueError(f"Unsafe file path: {f}")

    with tarfile.open(tar_file_name, "w:gz") as tar:
        for file in file_list:
            tar.add(file, arcname=file.name)


def get_log_excerpt(log_file, n_last_lines=30):
    """
    Get an excerpt from a log file, namely the n_last_lines of the file.

    Parameters
    ----------
    log_file: str or Path
        Log file to get the excerpt from.
    n_last_lines: int
        Number of last lines of the file to get.

    Returns
    -------
    str
        Excerpt from log file with header/footer
    """
    return (
        "\n\nRuntime error - See below the relevant part of the log/err file.\n\n"
        f"{log_file}\n"
        "====================================================================\n\n"
        f"{collect_final_lines(log_file, n_last_lines)}\n\n"
        "====================================================================\n"
    )


def get_file_age(file_path):
    """Get the age of a file in seconds since the last modification."""
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"'{file_path}' does not exist or is not a file.")

    file_stats = Path(file_path).stat()
    modification_time = file_stats.st_mtime
    current_time = time.time()

    return (current_time - modification_time) / 60


def _process_dict_keys(input_dict, case_func):
    """
    Process dictionary keys recursively.

    Parameters
    ----------
    input_dict: dict
        Dictionary to be processed.
    case_func: function
        Function to change case of keys (e.g., str.lower, str.upper).

    Returns
    -------
    dict
        Processed dictionary with keys changed.
    """
    output_dict = {}
    for key, value in input_dict.items():
        processed_key = case_func(key)
        if isinstance(value, dict):
            output_dict[processed_key] = _process_dict_keys(value, case_func)
        elif isinstance(value, list):
            processed_list = [
                _process_dict_keys(item, case_func) if isinstance(item, dict) else item
                for item in value
            ]
            output_dict[processed_key] = processed_list
        else:
            output_dict[processed_key] = value
    return output_dict


def change_dict_keys_case(data_dict, lower_case=True):
    """
    Change keys of a dictionary to lower or upper case recursively.

    Parameters
    ----------
    data_dict: dict
        Dictionary to be converted.
    lower_case: bool
        Change keys to lower (upper) case if True (False).

    Returns
    -------
    dict
        Dictionary with keys converted to lower or upper case.
    """
    # Determine which case function to use
    case_func = str.lower if lower_case else str.upper

    try:
        return _process_dict_keys(data_dict, case_func)
    except AttributeError as exc:
        _logger.error(f"Input is not a proper dictionary: {data_dict}")
        raise AttributeError from exc


def remove_substring_recursively_from_dict(data_dict, substring="\n"):
    """
    Remove substrings from all strings in a dictionary.

    Recursively crawls through the dictionary This e.g., allows to remove all newline characters
    from a dictionary.

    Parameters
    ----------
    data_dict: dict
        Dictionary to be converted.
    substring: str
        Substring to be removed.

    Raises
    ------
    AttributeError:
        if input is not a proper dictionary.
    """
    try:
        for key, value in data_dict.items():
            if isinstance(value, str):
                data_dict[key] = value.replace(substring, "")
            elif isinstance(value, list):
                modified_items = [
                    item.replace(substring, "") if isinstance(item, str) else item for item in value
                ]
                modified_items = [
                    (
                        remove_substring_recursively_from_dict(item, substring)
                        if isinstance(item, dict)
                        else item
                    )
                    for item in modified_items
                ]
                data_dict[key] = modified_items
            elif isinstance(value, dict):
                data_dict[key] = remove_substring_recursively_from_dict(value, substring)
    except AttributeError:
        _logger.debug(f"Input is not a dictionary: {data_dict}")
    return data_dict


def user_confirm():
    """
    Ask the user to enter y or n (case-insensitive) on the command line.

    Returns
    -------
    bool:
        True if the answer is Y/y.

    """
    while True:
        try:
            answer = input("Is this OK? [y/n]").lower()
            return answer == "y"
        except EOFError:
            break
    return False


def _get_value_dtype(value):
    """
    Get the data type of the given value.

    Parameters
    ----------
        Value to determine the data type.

    Returns
    -------
    type:
        Data type of the value.
    """
    if isinstance(value, (list | np.ndarray)):
        value = np.array(value)
        return value.dtype

    return type(value)


def validate_data_type(reference_dtype, value=None, dtype=None, allow_subtypes=True):
    """
    Validate data type of value or type object against a reference data type.

    Allow to check for exact data type or allow subtypes (e.g. uint is accepted for int).
    Take into account 'file' type as used in the model parameter database.

    Parameters
    ----------
    reference_dtype: str
        Reference data type to be checked against.
    value: any, optional
        Value to be checked (if dtype is None).
    dtype: type, optional
        Type object to be checked (if value is None).
    allow_subtypes: bool, optional
        If True, allow subtypes to be accepted.

    Returns
    -------
    bool:
        True if the data type is valid.
    """
    if value is None and dtype is None:
        raise ValueError("Either value or dtype must be given.")

    if value is not None and dtype is None:
        dtype = _get_value_dtype(value)

    # Strict comparison
    if not allow_subtypes:
        return np.issubdtype(dtype, reference_dtype)

    # Allow any sub-type of integer or float for success
    if (np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, "object")) and reference_dtype in (
        "string",
        "str",
        "file",
    ):
        return True

    if reference_dtype in ("boolean", "bool"):
        return _is_valid_boolean_type(dtype, value)

    return _is_valid_numeric_type(dtype, reference_dtype)


def _is_valid_boolean_type(dtype, value):
    """Check if dtype or value is a valid boolean type."""
    if value in {0, 1}:
        return True
    return np.issubdtype(dtype, np.bool_)


def _is_valid_numeric_type(dtype, reference_dtype):
    """Check if dtype is a valid numeric type compared to reference_dtype."""
    if np.issubdtype(dtype, np.integer):
        return np.issubdtype(reference_dtype, np.integer) or np.issubdtype(
            reference_dtype, np.floating
        )

    if np.issubdtype(dtype, np.floating):
        return np.issubdtype(reference_dtype, np.floating)

    return False


def convert_list_to_string(data, comma_separated=False, shorten_list=False, collapse_list=False):
    """
    Convert arrays to string (if required).

    Parameters
    ----------
    data: object
        Object of data to convert (e.g., double or list)
    comma_separated: bool
        If True, returns elements as a comma-separated string (default is space-separated).
    shorten_list: bool
        If True and all elements in the list are identical, returns a summary string
        like "all: value".  This is useful to make the configuration files more readable.
    collapse_list: bool
        If True and all elements in the list are identical, returns a single value
        instead of the entire list.

    Returns
    -------
    object or str:
        Converted data as string (if required)

    """
    if data is None or not isinstance(data, list | np.ndarray):
        return data
    if shorten_list and len(data) > 10 and all(np.isclose(item, data[0]) for item in data):
        return f"all: {data[0]}"
    if collapse_list and len(sorted(set(data))) == 1:
        data = [data[0]]
    if comma_separated:
        return ", ".join(str(item) for item in data)
    return " ".join(str(item) for item in data)


def convert_string_to_list(data_string, is_float=True, force_comma_separation=False):
    """
    Convert string (as used e.g. in sim_telarray) to list.

    Allow coma or space separated strings.

    Parameters
    ----------
    data_string: object
        String to be converted
    is_float: bool
        If True, convert to float, otherwise to int.
    force_comma_separation: bool
        If True, force comma separation.

    Returns
    -------
    list, str
        Converted data from string (if required).
        Return data_string if conversion fails.

    """
    try:
        if is_float:
            return [float(v) for v in data_string.split()]
        return [int(v) for v in data_string.split()]
    except ValueError:
        pass
    if "," in data_string:
        result = data_string.split(",")
        return [item.strip() for item in result]
    if " " in data_string and not force_comma_separation:
        return data_string.split()
    return data_string


def get_structure_array_from_table(table, column_names):
    """
    Get a structured array from an astropy table for a selected list of columns.

    Parameters
    ----------
    table: astropy.table.Table
        Table to be converted.
    column_names: list
        List of column names to be included in the structured array.

    Returns
    -------
    numpy.ndarray
        Structured array containing the table data.
    """
    return np.array(
        list(zip(*[np.array(table[col]) for col in column_names if col in table.colnames])),
        dtype=[(col, np.array(table[col]).dtype) for col in column_names if col in table.colnames],
    )


def convert_keys_in_dict_to_lowercase(data):
    """
    Recursively convert all dictionary keys to lowercase.

    Parameters
    ----------
    data: dict
        Dictionary to be converted.

    Returns
    -------
    dict
        Dictionary with all keys converted to lowercase.
    """
    if isinstance(data, dict):
        return {k.lower(): convert_keys_in_dict_to_lowercase(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_keys_in_dict_to_lowercase(i) for i in data]
    return data


def remove_key_from_dict(data, key_to_remove):
    """
    Remove a specific key from a dictionary recursively.

    Parameters
    ----------
    data: dict
        Dictionary to be processed.
    key_to_remove: str
        Key to be removed from the dictionary.

    Returns
    -------
    dict
        Dictionary with the specified key removed.
    """
    if isinstance(data, dict):
        return {
            k: remove_key_from_dict(v, key_to_remove) for k, v in data.items() if k != key_to_remove
        }
    if isinstance(data, list):
        return [remove_key_from_dict(i, key_to_remove) for i in data]
    return data


def _find_differences_dict(obj1, obj2, path, diffs):
    """Recursively find differences between two dictionaries."""
    for key in sorted(set(obj1) | set(obj2)):
        subpath = f"{path}['{key}']" if path else f"['{key}']"
        if key not in obj1:
            diffs.append(f"{subpath}: added in second object")
        elif key not in obj2:
            diffs.append(f"{subpath}: removed in second object")
        else:
            diffs.extend(find_differences_in_json_objects(obj1[key], obj2[key], subpath))


def find_differences_in_json_objects(obj1, obj2, path=""):
    """
    Recursively find differences between two JSON-like objects.

    Parameters
    ----------
    obj1: dict, list, or any
        First object to compare.
    obj2: dict, list, or any
        Second object to compare.
    path: str
        Path to the current object in the JSON structure, used for reporting differences.

    Returns
    -------
    list
        List of differences found between the two objects, with paths indicating where the
        differences occur.
    """
    diffs = []

    if not isinstance(obj1, type(obj2)):
        diffs.append(f"{path}: type changed from {type(obj1).__name__} to {type(obj2).__name__}")
        return diffs

    if isinstance(obj1, dict):
        _find_differences_dict(obj1, obj2, path, diffs)

    elif isinstance(obj1, list):
        if len(obj1) != len(obj2):
            diffs.append(f"{path}: list length changed from {len(obj1)} to {len(obj2)}")
        for i, (a, b) in enumerate(zip(obj1, obj2)):
            subpath = f"{path}[{i}]" if path else f"[{i}]"
            diffs.extend(find_differences_in_json_objects(a, b, subpath))

    elif obj1 != obj2:
        diffs.append(f"{path}: value changed from {obj1} to {obj2}")

    return diffs


def clear_default_sim_telarray_cfg_directories(command):
    """Prefix the command to clear default sim_telarray configuration directories.

    Parameters
    ----------
    command: str
        Command to be prefixed.

    Returns
    -------
    str
        Prefixed command.

    """
    return f"SIM_TELARRAY_CONFIG_PATH='' {command}"


def get_list_of_files_from_command_line(file_names, suffix_list):
    """
    Get a list of files from the command line.

    Files can be given as a list of file names or as a text file containing the list of files.
    The list of suffixes restrict the files types to be returned. Note that a file list must
    have a different suffix than those in the suffix list.

    Parameters
    ----------
    file_names: list
        List of file names to be checked.
    suffix_list: list
        List of suffixes to be checked.

    Returns
    -------
    list
        List of files with the given suffixes.
    """
    _files = []
    for one_file in file_names:
        path = Path(one_file)
        try:
            if path.suffix in suffix_list:
                _files.append(one_file)
            elif len(file_names) == 1:
                with open(one_file, encoding="utf-8") as file:
                    _files.extend(line.strip() for line in file)
        except FileNotFoundError as exc:
            _logger.error(f"{one_file} is not a file.")
            raise FileNotFoundError from exc
    return _files


def now_date_time_in_isoformat():
    """Return date and time in isoformat and second accuracy."""
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
