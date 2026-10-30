"""Value and quantity conversion."""

import collections.abc
import json
import logging
import re

import numpy as np
from astropy import units as u

_logger = logging.getLogger(__name__)

_DIMENSIONLESS_UNITS = (None, "", "dimensionless", "null")


def is_dimensionless_unit(unit):
    """
    Return True if unit encodes a dimensionless quantity.

    Parameters
    ----------
    unit: str, object, or None
        Unit string or Astropy unit object to be checked.

    Returns
    -------
    bool        True if the unit encodes a dimensionless quantity, False otherwise.
    """
    if unit is None:
        return True
    to_string = getattr(unit, "to_string", None)
    if to_string is not None:
        unit = to_string()
    return isinstance(unit, str) and unit in _DIMENSIONLESS_UNITS


def normalize_dimensionless_unit(unit):
    """
    Normalize dimensionless unit markers to None.

    Parameters
    ----------
    unit: str, object, or None
        Unit string or Astropy unit object to be normalized.

    Returns
    -------
    str, object, or None
        Normalized unit, where dimensionless units are converted to None.

    """
    if isinstance(unit, list | np.ndarray):
        return [normalize_dimensionless_unit(entry) for entry in unit]
    return None if is_dimensionless_unit(unit) else unit


def normalize_model_parameter_unit(value, unit):
    """Normalize a model parameter unit for use by model repository consumers.

    Parameters
    ----------
    value : object
        Model parameter value associated with the unit.
    unit : str, list, numpy.ndarray, or None
        Unit representation to normalize.

    Returns
    -------
    str, list, or None
        Normalized unit. Scalar units are canonicalized for scalar values and
        retained for list-valued parameters.
    """
    unit = normalize_dimensionless_unit(unit)
    if unit is None or isinstance(value, list | np.ndarray) or isinstance(unit, list):
        return unit
    return u.Unit(unit).to_string()


def extract_type_of_value(value):
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
        except TypeError, ValueError:
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

    return base_value, _unit_as_string(base_unit), base_type


def split_value_and_unit(value, is_integer=False):
    """
    Split a value into its value and unit.

    Takes into account the case where the value is a Quantity, a number,
    or a simtools-type string encoding a list of values and units.

    Parameters
    ----------
    value: str, int, float, bool, u.Quantity
        Value to be parsed.
    is_integer: bool
        Flag to indicate if the value is an integer.

    Returns
    -------
    value, str
        Value and units as (value, unit), or lists of values and unites
    """
    if isinstance(value, u.Quantity):
        return _split_value_is_quantity(value, is_integer)
    if isinstance(value, str):
        return _split_value_is_string(value, is_integer)
    if isinstance(value, list | np.ndarray):
        return _split_value_is_list(value, is_integer)
    return value, None


def _split_value_is_quantity(value, is_integer=False):
    """Split value and unit for an astropy Quantity."""
    if isinstance(value.value, list | np.ndarray):  # type [100.0, 200] * u.m
        return [int(v) if is_integer else v for v in value.value], [str(value.unit)] * len(value)
    return int(value.value) if is_integer else value.value, str(value.unit)


def _split_value_is_string(value, is_integer=False):
    """Split value and unit for a string."""
    if value.isdigit():  # single integer value
        return int(value), None
    try:  # single value with/without unit
        quantity = u.Quantity(value)
        unit = str(quantity.unit)
        if unit.isdigit():  # cases where numbers are wrongly identified as units
            raise ValueError
        return (int(quantity.value), unit) if is_integer else (quantity.value, unit)
    except ValueError:
        return _split_value_is_list(convert_string_to_list(value), is_integer)
    except TypeError:  # string value (not numerical)
        return value, None


def _split_value_is_list(value, is_integer=False):
    """Split value and unit for a list."""
    value_list = []
    unit_list = []
    for item in value:
        _value, _unit = split_value_and_unit(item)
        value_list.append(int(_value) if is_integer else _value)
        if isinstance(_unit, str):
            unit_list.append(_unit)
        else:
            unit_list.append(None)
    return value_list, unit_list


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
    ValueError
        If the value cannot be converted to the given unit.
    """
    unit = u.dimensionless_unscaled if is_dimensionless_unit(unit) else u.Unit(unit)
    if isinstance(value, str):
        if unit == u.dimensionless_unscaled:
            return value
        return u.Quantity(value).to(unit)

    if (
        isinstance(value, collections.abc.Iterable)
        and not isinstance(value, str)
        and not hasattr(value, "to")
    ):
        return [get_value_as_quantity(v, unit) for v in value]

    if hasattr(value, "to"):
        return value.to(unit)
    return u.Quantity(value, unit)


def _unit_as_string(unit):
    """Return the string representation of a unit. Collapse if it is a list of identical items."""
    if not isinstance(unit, list):
        unit = [unit]
    unit = [str(element) if element is not None else None for element in unit]
    return unit[0] if len(set(unit)) == 1 else unit


def get_value_in_unit(value, unit=None):
    """
    Return the numerical value of a Quantity in the requested unit.

    Parameters
    ----------
    value: astropy.units.Quantity or any
        Value to be converted.
    unit: astropy.units.Unit or str or None
        Unit to convert the value to.

    Returns
    -------
    float or any
        Numerical value of the Quantity in the requested unit,
        or the original value if it is not a Quantity and no target unit is requested.

    Raises
    ------
    TypeError
        If ``unit`` is requested but ``value`` is neither a Quantity nor a numeric scalar.
    """
    if isinstance(value, u.Quantity):
        return value.to_value(unit) if unit is not None else value.value

    if unit is not None and not isinstance(value, int | float):
        raise TypeError(
            f"Expected Quantity or numeric scalar when unit='{unit}', got {type(value)}"
        )

    return value


def ensure_list(value):
    """
    Return input value as list.

    Parameters
    ----------
    value: any
        Input value to be converted to a list.

    Returns
    -------
    list
        Converted value as list.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass
        return [stripped]
    return [value]


def ensure_string_lists(value):
    """
    Ensure the input value is returned as a list of strings.

    Parameters
    ----------
    value : any
        Input value to convert to a list of strings.

    Returns
    -------
    list or None
        List of strings or None if the input is None.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


def parse_typed_sequence(value, cast=float):
    """Parse scalar or sequence input into a list of typed values.

    Parameters
    ----------
    value : any
        Input value to parse. Supported forms are ``None``, scalar values,
        ``list``/``tuple``, or comma-separated strings.
    cast : callable, optional
        Conversion callable applied to each parsed item. Defaults to ``float``.

    Returns
    -------
    list
        List of converted values. Returns an empty list when ``value`` is ``None``.
    """

    def _cast_item(item):
        if cast is int:
            try:
                return int(item)
            except (TypeError, ValueError) as exc:
                float_value = float(item)
                if not float_value.is_integer():
                    raise ValueError(
                        f"Cannot safely cast non-integer value '{item}' to int"
                    ) from exc
                return int(float_value)
        return cast(item)

    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [_cast_item(item) for item in value]
    if isinstance(value, str) and "," in value:
        return [_cast_item(item.strip()) for item in value.split(",") if item.strip()]
    return [_cast_item(value)]


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
    Take into account 'file' type as used in the model parameter repository.

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

    if reference_dtype == "dict":
        return isinstance(value, dict)

    if reference_dtype == "list":
        return isinstance(value, list)

    return _is_valid_numeric_type(dtype, reference_dtype)


def _is_valid_boolean_type(dtype, value):
    """Check if dtype or value is a valid boolean type."""
    if value in {0, 1}:
        return True
    return np.issubdtype(dtype, np.bool_)


def _is_valid_numeric_type(dtype, reference_dtype):
    """Check if dtype is a valid numeric type compared to reference_dtype."""
    # Handle string reference_dtype (not numeric types)
    if reference_dtype in ("string", "str", "file"):
        return False

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


def replace_placeholders_recursively(data, replacements):
    """Replace placeholders recursively in strings nested in dicts/lists.

    Parameters
    ----------
    data : dict, list, str, or object
        Input structure to process.
    replacements : dict
        Mapping of placeholder strings to replacement strings.

    Returns
    -------
    dict, list, str, or object
        Processed copy with placeholders replaced in string values.
    """
    if isinstance(data, dict):
        return {
            key: replace_placeholders_recursively(value, replacements)
            for key, value in data.items()
        }
    if isinstance(data, list):
        return [replace_placeholders_recursively(item, replacements) for item in data]
    if isinstance(data, str):
        for placeholder, replacement in replacements.items():
            data = data.replace(placeholder, replacement)
    return data
