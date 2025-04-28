"""Custom format checkers for jsonschema validation."""

import re

import astropy.units as u
import jsonschema

from simtools.corsika.primary_particle import PrimaryParticle
from simtools.utils import names

format_checker = jsonschema.FormatChecker()


@format_checker.checks("astropy_unit")
def check_astropy_unit(unit_string):
    """Validate astropy units (including dimensionless) for jsonschema."""
    try:
        u.Unit(unit_string)
    except (ValueError, TypeError) as exc:
        if unit_string != "dimensionless":
            raise ValueError(f"'{unit_string}' is not a valid Unit") from exc
    return True


@format_checker.checks("astropy_unit_of_time")
def check_astropy_unit_of_time(unit_string):
    """Validate astropy units that this is an astropy unit of time."""
    u.Unit(unit_string).to("s")
    return True


@format_checker.checks("astropy_unit_of_length)")
def check_astropy_unit_of_length(unit_string):
    """Validate astropy units that this is an astropy unit of length."""
    u.Unit(unit_string).to("m")
    return True


@format_checker.checks("array_element")
def check_array_element(element):
    """Validate array elements for jsonschema."""
    names.validate_array_element_name(element)
    return True


@format_checker.checks("array_triggers_name")
def check_array_triggers_name(name):
    """Validate array trigger names for jsonschema."""
    pattern = r"(.*)(?=_single_telescope|_array)"
    if not re.match(pattern, name):
        raise ValueError(f"Array trigger name '{name}' does not match pattern '{pattern}'")
    names.validate_array_element_type(re.match(pattern, name).group(1))
    return True


@format_checker.checks("common_particle_name")
def check_common_particle_name(name):
    """Validate common particle names for jsonschema."""
    if name in PrimaryParticle.particle_names() or name == "default":
        return True
    raise ValueError(f"Invalid common particle name: '{name}'")
