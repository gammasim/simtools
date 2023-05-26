import copy
import logging
import mmap
import os
import re
from collections import namedtuple
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates.errors import UnitsError
from astropy.io.misc import yaml

__all__ = [
    "collect_data_from_yaml_or_dict",
    "collect_final_lines",
    "collect_kwargs",
    "InvalidConfigData",
    "InvalidConfigEntry",
    "MissingRequiredConfigEntry",
    "UnableToIdentifyConfigEntry",
    "get_log_level_from_user",
    "rotate",
    "separate_args_and_config_data",
    "set_default_kwargs",
    "sort_arrays",
    "validate_config_data",
    "get_log_excerpt",
]


class UnableToIdentifyConfigEntry(Exception):
    """Exception for unable to indentify configuration entry."""


class MissingRequiredConfigEntry(Exception):
    """Exception for missing required configuration entry."""


class InvalidConfigEntry(Exception):
    """Exception for invalid configuration entry."""


class InvalidConfigData(Exception):
    """Exception for invalid configuration data."""


def file_has_text(file, text):
    """
    Check whether a file contain a certain piece of text.

    Parameters
    ----------
    file: str
        Path of the file.
    text: str
        Piece of text to be searched for.

    Returns
    -------
    bool
        True if file has text.
    """
    with open(file, "rb", 0) as string_file, mmap.mmap(
        string_file.fileno(), 0, access=mmap.ACCESS_READ
    ) as text_file_input:
        re_search_1 = re.compile(f"{text}".encode())
        search_result_1 = re_search_1.search(text_file_input)
        if search_result_1 is None:
            return False

        return True


def validate_config_data(config_data, parameters):
    """
    Validate a generic config_data dict by using the info
    given by the parameters dict. The entries will be validated
    in terms of length, units and names.

    See data/test-data/test_parameters.yml for an example of the structure
    of the parameters dict.

    Parameters
    ----------
    config_data: dict
        Input config data.
    parameters: dict
        Parameter information necessary for validation.

    Raises
    ------
    UnableToIdentifyConfigEntry
        When an entry in config_data cannot be identified among the parameters.
    MissingRequiredConfigEntry
        When a parameter without default value is not given in config_data.
    InvalidConfigEntry
        When an entry in config_data is invalid (wrong len, wrong unit, ...).

    Returns
    -------
    namedtuple:
        Containing the validated config data entries.
    """

    logger = logging.getLogger(__name__)

    # Dict to be filled and returned
    out_data = {}

    if config_data is None:
        config_data = {}

    # Collecting all entries given as in config_data.
    for key_data, value_data in config_data.items():

        is_identified = False
        # Searching for the key in the parameters.
        for par_name, par_info in parameters.items():
            names = par_info.get("names", [])
            if key_data != par_name and key_data.lower() not in [n.lower() for n in names]:
                continue
            # Matched parameter
            validated_value = _validate_and_convert_value(par_name, par_info, value_data)
            out_data[par_name] = validated_value
            is_identified = True

        # Raising error for an unidentified input.
        if not is_identified:
            msg = f"Entry {key_data} in config_data cannot be identified."
            logger.error(msg)
            raise UnableToIdentifyConfigEntry(msg)

    # Checking for parameters with default option.
    # If it is not given, filling it with the default value.
    for par_name, par_info in parameters.items():
        if par_name in out_data:
            continue
        if "default" in par_info.keys() and par_info["default"] is not None:
            validated_value = _validate_and_convert_value(par_name, par_info, par_info["default"])
            out_data[par_name] = validated_value
        elif "default" in par_info.keys() and par_info["default"] is None:
            out_data[par_name] = None
        else:
            msg = (
                f"Required entry in config_data {par_name} " + "was not given (there may be more)."
            )
            logger.error(msg)
            raise MissingRequiredConfigEntry(msg)

    configuration_data = namedtuple("configuration_data", out_data)
    return configuration_data(**out_data)


def _validate_and_convert_value_without_units(value, value_keys, par_name, par_info):
    """
    Validate input user parameter for input values without units.

    Parameters
    ----------
    value: list
       list of user input values.
    value_keys: list
       list of keys if user input was a dict; otherwise None.
    par_name: str
       name of parameter.
    par_info: dict
        dictionary with parameter info.

    Returns
    -------
    list, dict
        validated and converted input data

    """
    logger = logging.getLogger(__name__)

    _, undefined_length = _check_value_entry_length(value, par_name, par_info)

    # Checking if values have unit and raising error, if so.
    if all(isinstance(v, str) for v in value):
        # In case values are string, e.g. mirror_numbers = 'all'
        # This is needed otherwise the elif condition will break
        pass
    elif any(u.Quantity(v).unit != u.dimensionless_unscaled for v in value):
        msg = f"Config entry {par_name} should not have units"
        logger.error(msg)
        raise InvalidConfigEntry(msg)

    if value_keys:
        return dict(zip(value_keys, value))
    return value if len(value) > 1 or undefined_length else value[0]


def _check_value_entry_length(value, par_name, par_info):
    """
    Validate length of user input parmeters

    Parameters
    ----------
    value: list
        list of user input values
    par_name: str
        name of parameter
    par_info: dict
        dictionary with parameter info

    Returns
    -------
    value_length: int
        length of input list
    undefined_length: bool
        state of input list

    """
    logger = logging.getLogger(__name__)

    # Checking the entry length
    value_length = len(value)
    logger.debug(f"Value len of {par_name}: {value_length}")
    undefined_length = False
    try:
        if par_info["len"] is None:
            undefined_length = True
        elif value_length != par_info["len"]:
            msg = f"Config entry with wrong len: {par_name}"
            logger.error(msg)
            raise InvalidConfigEntry(msg)
    except KeyError:
        logger.error("Missing len entry in par_info")
        raise

    return value_length, undefined_length


def _validate_and_convert_value_with_units(value, value_keys, par_name, par_info):
    """
    Validate input user parameter for input values with units.

    Parameters
    ----------
    value: list
       list of user input values
    value_keys: list
       list of keys if user input was a dict; otherwise None
    par_name: str
       name of parameter

    Returns
    -------
    list, dict
        validated and converted input data

    """
    logger = logging.getLogger(__name__)

    value_length, undefined_length = _check_value_entry_length(value, par_name, par_info)

    par_unit = copy_as_list(par_info["unit"])

    if undefined_length and len(par_unit) != 1:
        msg = f"Config entry with undefined length should have a single unit: {par_name}"
        logger.error(msg)
        raise InvalidConfigEntry(msg)
    if len(par_unit) == 1:
        par_unit *= value_length

    # Checking units and converting them, if needed.
    value_with_units = list()
    for arg, unit in zip(value, par_unit):
        # In case a entry is None, None should be returned.
        if unit is None or arg is None:
            value_with_units.append(arg)
            continue

        # Converting strings to Quantity
        if isinstance(arg, str):
            arg = u.quantity.Quantity(arg)

        if not isinstance(arg, u.quantity.Quantity):
            msg = f"Config entry given without unit: {par_name}"
            logger.error(msg)
            raise InvalidConfigEntry(msg)
        if not arg.unit.is_equivalent(unit):
            msg = f"Config entry given with wrong unit: {par_name}"
            logger.error(msg)
            raise InvalidConfigEntry(msg)
        value_with_units.append(arg.to(unit).value)

    if value_keys:
        return dict(zip(value_keys, value_with_units))

    return (
        value_with_units if len(value_with_units) > 1 or undefined_length else value_with_units[0]
    )


def _validate_and_convert_value(par_name, par_info, value_in):
    """
    Validate input user parameter and convert it to the right units, if needed.
    Returns the validated arguments in a list.
    """

    if isinstance(value_in, dict):
        value = [d for (k, d) in value_in.items()]
        value_keys = [k for (k, d) in value_in.items()]
    else:
        value = copy_as_list(value_in)
        value_keys = None

    if "unit" not in par_info.keys():
        return _validate_and_convert_value_without_units(value, value_keys, par_name, par_info)

    return _validate_and_convert_value_with_units(value, value_keys, par_name, par_info)


def collect_data_from_yaml_or_dict(in_yaml, in_dict, allow_empty=False):
    """
    Collect input data that can be given either as a dict or as a yaml file.

    Parameters
    ----------
    in_yaml: str
        Name of the yaml file.
    in_dict: dict
        Data as dict.
    allow_empty: bool
        If True, an error won't be raised in case both yaml and dict are None.

    Returns
    -------
    data: dict
        Data as dict.
    """
    _logger = logging.getLogger(__name__)

    if in_yaml is not None:
        if in_dict is not None:
            _logger.warning("Both in_dict in_yaml were given - in_yaml will be used")
        with open(in_yaml) as file:
            data = yaml.load(file)
        return data
    if in_dict is not None:
        return dict(in_dict)

    msg = "config_data has not been provided (by yaml file neither by dict)"
    if allow_empty:
        _logger.debug(msg)
        return None

    _logger.error(msg)
    raise InvalidConfigData(msg)


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
    out_kwargs = dict()
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


def sort_arrays(*args):
    """Sort arrays

    Parameters
    ----------
    *args
        Arguments to be sorted.
    Returns
    -------
    list
        Sorted args.
    """

    order_array = copy.copy(args[0])
    new_args = list()
    for arg in args:
        _, value = zip(*sorted(zip(order_array, arg)))
        new_args.append(list(value))
    return new_args


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
    with open(file, "rb") as read_obj:
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
        "warn": logging.WARNING,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    log_level_lower = log_level.lower()
    if log_level_lower not in possible_levels:
        raise ValueError(
            f"'{log_level}' is not a logging level, "
            f"only possible ones are {list(possible_levels.keys())}"
        )

    return possible_levels[log_level_lower]


def copy_as_list(value):
    """
    Copy value and, if it is not a list, turn it into a list with a single entry.

    Parameters
    ----------
    value single variable of any type or list

    Returns
    -------
    value: list
        Copy of value if it is a list of [value] otherwise.
    """
    if isinstance(value, str):
        return [value]

    try:
        return list(value)
    except Exception:
        return [value]


def separate_args_and_config_data(expected_args, **kwargs):
    """
    Separate kwargs into the arguments expected for instancing a class and the dict to be given as\
    config_data. This function is specific for methods from_kwargs in classes which use the \
    validate_config_data system.

    Parameters
    ----------
    expected_args: list of str
        List of arguments expected for the class.
    **kwargs

    Returns
    -------
    dict, dict
        A dict with the args collected and another one with config_data.
    """
    args = dict()
    config_data = dict()
    for key, value in kwargs.items():
        if key in expected_args:
            args[key] = value
        else:
            config_data[key] = value

    return args, config_data


def program_is_executable(program):
    """
    Checks if program exists and is executable

    Follows https://stackoverflow.com/questions/377017/

    """

    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, _ = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        try:
            for path in os.environ["PATH"].split(os.pathsep):
                exe_file = os.path.join(path, program)
                if is_exe(exe_file):
                    return exe_file
        except KeyError:
            return None

    return None


def find_file(name, loc):
    """
    Search for files inside of given directories, recursively, and return its full path.

    Parameters
    ----------
    name: str
        File name to be searched for.
    loc: Path
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
    _logger = logging.getLogger(__name__)

    all_locations = copy.copy(loc)
    all_locations = [all_locations] if not isinstance(all_locations, list) else all_locations

    def _search_directory(directory, filename, rec=False):
        if not Path(directory).exists():
            msg = f"Directory {directory} does not exist"
            _logger.debug(msg)
            return None

        file = Path(directory).joinpath(filename)
        if file.exists():
            _logger.debug(f"File {filename} found in {directory}")
            return file
        if not rec:  # Not recursively
            return None

        for subdir in Path(directory).iterdir():
            if not subdir.is_dir():
                continue
            file = _search_directory(subdir, filename, True)
            if file is not None:
                return file
        return None

    # Searching file locally
    file = _search_directory(".", name)
    if file is not None:
        return file
    # Searching file in given locations
    for location_now in all_locations:
        file = _search_directory(location_now, name, True)
        if file is not None:
            return file
    msg = f"File {name} could not be found in {all_locations}"
    _logger.error(msg)
    raise FileNotFoundError(msg)


def change_dict_keys_case(data_dict, lower_case=True):
    """
    Change keys of a dictionary to lower or upper case. Crawls through the dictionary and changes\
    all keys. Takes into account list of dictionaries, as e.g. found in the top level data model.

    Parameters
    ----------
    data_dict: dict
        Dictionary to be converted.
    lower_case: bool
        Change keys to lower (upper) case if True (False).
    """

    _return_dict = {}
    try:
        for key in data_dict.keys():
            if lower_case:
                _key_changed = key.lower()
            else:
                _key_changed = key.upper()
            if isinstance(data_dict[key], dict):
                _return_dict[_key_changed] = change_dict_keys_case(data_dict[key], lower_case)
            elif isinstance(data_dict[key], list):
                _tmp_list = []
                for _list_entry in data_dict[key]:
                    if isinstance(_list_entry, dict):
                        _tmp_list.append(change_dict_keys_case(_list_entry, lower_case))
                    else:
                        _tmp_list.append(_list_entry)
                _return_dict[_key_changed] = _tmp_list
            else:
                _return_dict[_key_changed] = data_dict[key]
    except AttributeError:
        logger = logging.getLogger(__name__)
        logger.error(f"Invalid method argument: {data_dict}")
        raise

    return _return_dict


@u.quantity_input(rotation_angle=u.rad)
def rotate(rotation_angle, x, y):
    """
    Rotate x and y by the rotation angle given in rotation_angle, in radians (astropy.units.rad),
    or in degrees (astropy.units.deg).
    The function returns the rotated x and y values in the same unit given.
    The direction of rotation of the elements is counterclockwise.

    Parameters
    ----------
    rotation_angle: astropy.units.rad
        Angle to rotate the array in radians.
    x: numpy.array or list
        x positions of the telescopes, usually in meters.
    y: numpy.array or list
        y positions of the telescopes, usually in meters.

    Returns
    -------
    2-tuple of list
        x and y positions of the rotated telescopes positions.

    Raises
    ------
    TypeError:
        If type of x and y parameters are not valid.
    RuntimeError:
        If the length of x and y are different.
    UnitsError:
        If the unit of x and y are different.
    """
    allowed_types = (list, np.ndarray, u.Quantity, float, int)
    if not all(isinstance(variable, allowed_types) for variable in [x, y]):
        raise TypeError("x and y types are not valid! Cannot perform transformation.")

    if (
        np.sum(
            np.array([isinstance(x, type_now) for type_now in allowed_types[:-2]])
            * np.array([isinstance(y, type_now) for type_now in allowed_types[:-2]])
        )
        == 0
    ):
        raise TypeError("x and y are not from the same type! Cannot perform transformation.")

    if not isinstance(x, (list, np.ndarray)):
        x = [x]
    if not isinstance(y, (list, np.ndarray)):
        y = [y]

    if len(x) != len(y):
        raise RuntimeError(
            "Cannot perform coordinate transformation when x and y have different lengths."
        )
    if all(isinstance(variable, (u.Quantity)) for variable in [x, y]):
        if not isinstance(x[0].unit, type(y[0].unit)):
            raise UnitsError(
                "Cannot perform coordinate transformation when x and y have different units."
            )

    x_trans, y_trans = [np.zeros_like(x).astype(float) for i in range(2)]

    for step, _ in enumerate(x):
        x_trans[step] = x[step] * np.cos(rotation_angle) - y[step] * np.sin(rotation_angle)
        y_trans[step] = x[step] * np.sin(rotation_angle) + y[step] * np.cos(rotation_angle)
    return x_trans, y_trans


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
        "\n\nRuntime error - See below the relevant part of the log file.\n\n"
        f"{log_file}\n"
        "====================================================================\n\n"
        f"{collect_final_lines(log_file, n_last_lines)}\n\n"
        "====================================================================\n"
    )
