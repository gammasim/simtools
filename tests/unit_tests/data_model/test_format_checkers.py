#!/usr/bin/python3

import re

import pytest
from astropy.units.core import UnitConversionError

from simtools.data_model import format_checkers


def test_check_astropy_unit():
    assert format_checkers.check_astropy_unit("dimensionless")
    assert format_checkers.check_astropy_unit("m")
    assert format_checkers.check_astropy_unit("")
    with pytest.raises(ValueError, match="'None' is not a valid Unit"):
        format_checkers.check_astropy_unit(None)

    with pytest.raises(ValueError, match="'not a unit!' is not a valid Unit"):
        format_checkers.check_astropy_unit("not a unit!")


def test_check_astropy_unit_of_time():
    assert format_checkers.check_astropy_unit_of_time("ns")
    with pytest.raises(
        UnitConversionError, match=re.escape("'km' (length) and 's' (time) are not convertible")
    ):
        format_checkers.check_astropy_unit_of_time("km")

    with pytest.raises(TypeError, match=re.escape("None is not a valid Unit")):
        format_checkers.check_astropy_unit_of_time(None)


def test_check_astropy_unit_of_length():
    assert format_checkers.check_astropy_unit_of_length("km")
    with pytest.raises(
        UnitConversionError, match=re.escape("'ns' (time) and 'm' (length) are not convertible")
    ):
        format_checkers.check_astropy_unit_of_length("ns")

    with pytest.raises(TypeError, match=re.escape("None is not a valid Unit")):
        format_checkers.check_astropy_unit_of_length(None)


def test_check_array_elements():
    assert format_checkers.check_array_element("MSTN-15")

    with pytest.raises(ValueError, match=r"^Invalid name"):
        format_checkers.check_array_element("not_an_array_element")


def test_check_array_triggers_name():
    assert format_checkers.check_array_triggers_name("MSTN_array")
    assert format_checkers.check_array_triggers_name("MSTN_single_telescope")

    with pytest.raises(ValueError, match=r"^Invalid name"):
        format_checkers.check_array_triggers_name("not_an_array_trigger")

    with pytest.raises(ValueError, match=r"^Array trigger name"):
        format_checkers.check_array_triggers_name("MSTN")
