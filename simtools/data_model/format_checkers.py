"""Custom format checkers for jsonschema validation."""

import re

import astropy.units as u
import jsonschema

from simtools.utils import names

format_checker = jsonschema.FormatChecker()


@format_checker.checks("astropy_unit", raises=ValueError)
def check_astropy_unit(unit_string):
    """Validate astropy units (including dimensionless) for jsonschema."""
    try:
        u.Unit(unit_string)
        return True
    except (ValueError, TypeError):
        return unit_string == "dimensionless"


@format_checker.checks("array_element", raises=ValueError)
def check_array_element(element):
    """Validate array elements for jsonschema."""
    names.validate_array_element_name(element)
    return True


@format_checker.checks("array_trigger_name", raises=ValueError)
def check_array_trigger_name(name):
    """Validate array trigger names for jsonschema."""
    pattern = r"(.*)(?=_single_telescope|_array)"
    if not re.match(pattern, name):
        raise ValueError(f"Array trigger name '{name}' does not match pattern '{pattern}'")
    names.validate_array_element_type(re.match(pattern, name).group(1))
    return True
