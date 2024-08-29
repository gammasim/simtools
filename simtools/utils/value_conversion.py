"""Value and quantity conversion."""

import logging
import re
from collections import namedtuple

from astropy import units as u

import simtools.utils.general as gen

_logger = logging.getLogger(__name__)

__all__ = [
    "validate_config_data",
    "InvalidConfigEntryError",
    "MissingRequiredConfigEntryError",
    "UnableToIdentifyConfigEntryError",
]


class InvalidConfigEntryError(Exception):
    """Exception for invalid configuration entry."""


class UnableToIdentifyConfigEntryError(Exception):
    """Exception for unable to identify configuration entry."""


class MissingRequiredConfigEntryError(Exception):
    """Exception for missing required configuration entry."""


def extract_type_of_value(value) -> str:
    """
    Extract the string representation of the the type of a value.

    For example, for a string, it returns 'str' rather than '<class 'str'>'.
    Take into account also the case where the value is a numpy type.
    """
    _type = str(type(value))
    if "numpy" in _type:
        return re.sub(r"\d+", "", _type.split("'")[1].split(".")[-1])
    if "astropy" in _type:
        raise NotImplementedError("Astropy types are not supported yet.")
    return _type.split("'")[1]


def get_value_unit_type(value, unit_str=None):
    """
    Get the value, unit and type of a value.

    The value is stripped of its unit and the unit is returned
    in its string form (i.e., to_string()).
    The type is returned as a string representation of the type.
    For example, for a string, it returns 'str' rather than '<class 'str'>'.
    An additional unit string can be given and the return value is converted to this units.

    Note that Quantities are always floats, even if the original value is represented as an int.

    Parameters
    ----------
    value: str, int, float, bool, u.Quantity
        Value to be parsed.
    unit_str: str
        Unit to be used for the value.

    Returns
    -------
    type of value, str, str
        Value, unit in string representation (to_string())),
        and string representation of the type of the value.
    """
    base_unit = None
    if isinstance(value, str | u.Quantity):
        try:
            _quantity_value = u.Quantity(value)
            base_value = _quantity_value.value
            base_type = extract_type_of_value(base_value)
            if _quantity_value.unit.to_string() != "":
                base_unit = _quantity_value.unit.to_string()
                try:  # handle case of e.g., "0 0" and avoid unit.scale
                    float(base_unit)
                    base_value = value
                    base_type = "str"
                    base_unit = None
                except ValueError:
                    pass
        # ValueError: covers strings of type "5 not a unit"
        except (TypeError, ValueError):
            base_value = value
            base_type = "str"
    else:
        base_value = value
        base_type = extract_type_of_value(base_value)

    if unit_str is not None:
        try:
            base_value = base_value * u.Unit(base_unit).to(u.Unit(unit_str))
        except u.UnitConversionError:
            _logger.error(f"Cannot convert {base_unit} to {unit_str}.")
            raise
        except TypeError:
            pass
        base_unit = unit_str

    return base_value, base_unit, base_type


def get_value_as_quantity(value, unit):
    """
    Get a value as a Quantity with a given unit. If value is a Quantity, convert to the given unit.

    Parameters
    ----------
    value:
        value to get a unit. It can be a float, int, or a Quantity (convertible to 'unit').
    unit: astropy.units.Unit
        Unit to apply to 'quantity'.

    Returns
    -------
    astropy.units.Quantity
        Quantity of value 'quantity' and unit 'unit'.

    Raises
    ------
    u.UnitConversionError
        If the value cannot be converted to the given unit.
    """
    if isinstance(value, u.Quantity):
        try:
            return value.to(unit)
        except u.UnitConversionError:
            _logger.error(f"Cannot convert {value.unit} to {unit}.")
            raise
    return value * unit


def _process_default_value(par_name, par_info, out_data, _logger):
    """
    Process a default value for a parameter if it was not provided in config_data.

    Parameters
    ----------
    par_name: str
        Parameter name.
    par_info: dict
        Parameter information.
    out_data: dict
        Dictionary to store validated data.
    _logger: Logger
        Logger object for logging messages.

    Raises
    ------
    MissingRequiredConfigEntryError
        If a required parameter without default value is not given in config_data.
    """
    if "default" not in par_info:
        msg = f"Required entry in config_data {par_name} was not given."
        _logger.error(msg)
        raise MissingRequiredConfigEntryError(msg)

    default_value = par_info["default"]

    if default_value is None:
        out_data[par_name] = None
    else:
        if isinstance(default_value, dict):
            default_value = default_value["value"]

        if "unit" in par_info and not isinstance(default_value, u.Quantity):
            default_value *= u.Unit(par_info["unit"])

        validated_value = _validate_and_convert_value(par_name, par_info, default_value)
        out_data[par_name] = validated_value


def validate_config_data(config_data, parameters, ignore_unidentified=False, _logger=None):
    """
    Validate a generic config_data dict by using the info given by the parameters dict.

    The entries will be validated in terms of length, units and names.

    See ./tests/resources/test_parameters.yml for an example of the structure
    of the parameters dict.

    Parameters
    ----------
    config_data: dict
        Input config data.
    parameters: dict
        Parameter information necessary for validation.
    ignore_unidentified: bool, optional
        If set to True, unidentified parameters provided in config_data are ignored
        and a debug message is printed. Otherwise, an unidentified parameter leads to an error.
        Default is False.
    _logger: Logger, optional
        Logger object for logging messages. If not provided, defaults to printing to console.

    Raises
    ------
    UnableToIdentifyConfigEntryError
        When an entry in config_data cannot be identified among the parameters.
    MissingRequiredConfigEntryError
        When a parameter without default value is not given in config_data.
    InvalidConfigEntryError
        When an entry in config_data is invalid (wrong len, wrong unit, ...).

    Returns
    -------
    namedtuple:
        Containing the validated config data entries.
    """
    if _logger is None:
        _logger = logging.getLogger(__name__)

    out_data = {}

    if config_data is None:
        config_data = {}

    for key_data, value_data in config_data.items():
        is_identified = _process_identified_entry(parameters, key_data, value_data, out_data)

        if not is_identified:
            _handle_unidentified_entry(key_data, ignore_unidentified, _logger)

    for par_name, par_info in parameters.items():
        if par_name in out_data:
            continue

        _process_default_value(par_name, par_info, out_data, _logger)

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
    _, undefined_length = _check_value_entry_length(value, par_name, par_info)

    # Checking if values have unit and raising error, if so.
    if all(isinstance(v, str) for v in value):
        # In case values are string, e.g. mirror_numbers = 'all'
        # This is needed otherwise the elif condition will break
        pass
    elif any(u.Quantity(v).unit != u.dimensionless_unscaled for v in value):
        msg = f"Config entry {par_name} should not have units"
        _logger.error(msg)
        raise InvalidConfigEntryError(msg)

    if value_keys:
        return dict(zip(value_keys, value))
    return value if len(value) > 1 or undefined_length else value[0]


def _check_value_entry_length(value, par_name, par_info):
    """
    Validate length of user input parameters.

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
    # Checking the entry length
    value_length = len(value)
    _logger.debug(f"Value len of {par_name}: {value_length}")
    undefined_length = False
    try:
        if par_info["len"] is None:
            undefined_length = True
        elif value_length != par_info["len"]:
            msg = f"Config entry with wrong len: {par_name}"
            _logger.error(msg)
            raise InvalidConfigEntryError(msg)
    except KeyError:
        _logger.error("Missing len entry in par_info")
        raise

    return value_length, undefined_length


def _convert_to_valid_unit(arg, unit, par_name):
    """
    Convert argument to the valid unit.

    Parameters
    ----------
    arg: any
        The argument value to convert
    unit: str or None
        The unit to convert to
    par_name: str
        The parameter name for error messages

    Returns
    -------
    float or int
        The converted value

    Raises
    ------
    ValueError
        If the argument is not a valid quantity or has an incorrect unit.
    """
    # In case a entry is None, None should be returned.
    if unit is None or arg is None:
        return arg

    # Converting strings to Quantity
    if isinstance(arg, str):
        arg = u.Quantity(arg)

    if not isinstance(arg, u.Quantity):
        msg = f"Config entry given without unit: {par_name}"
        _logger.error(msg)
        raise InvalidConfigEntryError(msg)

    if not arg.unit.is_equivalent(unit):
        msg = f"Config entry given with wrong unit: {par_name} (should be {unit}, is {arg.unit})"
        _logger.error(msg)
        raise InvalidConfigEntryError(msg)

    return arg.to(unit).value


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
    par_info: dict
       parameter information including units

    Returns
    -------
    list, dict
        validated and converted input data

    Raises
    ------
    InvalidConfigEntryError
        If there are issues with unit validation or if the value entry length is undefined.
    """
    value_length, undefined_length = _check_value_entry_length(value, par_name, par_info)
    par_unit = gen.copy_as_list(par_info["unit"])

    if undefined_length and len(par_unit) != 1:
        msg = f"Config entry with undefined length should have a single unit: {par_name}"
        _logger.error(msg)
        raise InvalidConfigEntryError(msg)

    if len(par_unit) == 1:
        par_unit *= value_length
    # Checking units and converting them, if needed.
    value_with_units = [
        _convert_to_valid_unit(arg, unit, par_name) for arg, unit in zip(value, par_unit)
    ]

    if value_keys:
        return dict(zip(value_keys, value_with_units))

    return (
        value_with_units[0]
        if not undefined_length and len(value_with_units) == 1
        else value_with_units
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
        value = gen.copy_as_list(value_in)
        value_keys = None

    if "unit" not in par_info.keys():
        return _validate_and_convert_value_without_units(value, value_keys, par_name, par_info)

    return _validate_and_convert_value_with_units(value, value_keys, par_name, par_info)


def _handle_unidentified_entry(key_data, ignore_unidentified, _logger):
    """
    Handle an unidentified entry in config_data based on the ignore_unidentified flag.

    Parameters
    ----------
    key_data: str
        Key from config_data that cannot be identified.
    ignore_unidentified: bool
        If set to True, unidentified parameters provided in config_data are ignored
        and a debug message is printed. Otherwise, an unidentified parameter leads to an error.
    _logger: Logger
        Logger object for logging messages.

    Raises
    ------
    UnableToIdentifyConfigEntryError
        If ignore_unidentified is False and an unidentified parameter leads to an error.
    """
    msg = f"Entry {key_data} in config_data cannot be identified"
    if ignore_unidentified:
        _logger.debug(f"{msg}, ignoring.")
    else:
        _logger.error(f"{msg}, stopping.")
        raise UnableToIdentifyConfigEntryError(msg)


def _process_identified_entry(parameters, key_data, value_data, out_data):
    """
    Process an identified entry in config_data based on parameters and validate the value.

    Parameters
    ----------
    parameters: dict
        Parameter information necessary for validation.
    key_data: str
        Key from config_data.
    value_data:
        Value associated with the key_data in config_data.
    out_data: dict
        Dictionary to store validated data.

    Returns
    -------
    bool
        True if the entry was identified and processed successfully, False otherwise.

    Raises
    ------
    InvalidConfigEntryError
        If the value associated with the key_data is invalid based on parameter constraints.
    """
    is_identified = False
    for par_name, par_info in parameters.items():
        names = par_info.get("names", [])
        if key_data == par_name or key_data.lower() in map(str.lower, names):
            validated_value = _validate_and_convert_value(par_name, par_info, value_data)
            out_data[par_name] = validated_value
            is_identified = True
            break

    return is_identified
