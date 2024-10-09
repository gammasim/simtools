"""Custom format checkers for jsonschema validation."""

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
    except (ValueError, TypeError) as exc:
        if unit_string == "dimensionless":
            return True
        raise ValueError(f"'{unit_string}' is not a valid Unit") from exc


@format_checker.checks("array_element", raises=ValueError)
def check_array_element(element):
    """Validate array elements for jsonschema."""
    names.validate_array_element_name(element)
    return True
