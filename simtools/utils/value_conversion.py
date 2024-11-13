"""Value and quantity conversion."""

import logging
import re

import numpy as np
from astropy import units as u

import simtools.utils.general as gen

_logger = logging.getLogger(__name__)


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


def split_value_and_unit(value):
    """
    Split a value into its value and unit.

    Takes into account the case where the value is a Quantity, a number,
    or a simtools-type string encoding a list of values and units.
    """
    if isinstance(value, u.Quantity):
        if isinstance(value.value, list | np.ndarray):  # type [100.0, 200] * u.m,
            return value.value, [str(value.unit)] * len(value)
        return value.value, str(value.unit)
    if isinstance(value, str):
        if value.isdigit():  # single integer value
            return int(value), None
        try:  # single value with/without unit
            return u.Quantity(value).value, str(u.Quantity(value).unit)
        except ValueError:
            value = gen.convert_string_to_list(value)
    if isinstance(value, list | np.ndarray):
        value_list = []
        unit_list = []
        print("LL", value)
        for item in value:
            _value, _unit = split_value_and_unit(item)
            value_list.append(_value)
            if isinstance(_unit, str):
                unit_list.append(_unit)
            else:
                unit_list.append(None)
        return value_list, unit_list
    return value, None


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
