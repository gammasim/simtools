#!/usr/bin/python3

import pytest

from simtools.data_model import format_checkers


def test_check_astropy_unit():
    assert format_checkers.check_astropy_unit("dimensionless")
    assert format_checkers.check_astropy_unit("m")
    assert format_checkers.check_astropy_unit("")
    with pytest.raises(ValueError, match="'None' is not a valid Unit"):
        format_checkers.check_astropy_unit(None)

    with pytest.raises(ValueError, match="'not a unit!' is not a valid Unit"):
        format_checkers.check_astropy_unit("not a unit!")


def test_check_array_elements():
    assert format_checkers.check_array_element("MSTN-15")

    with pytest.raises(ValueError, match=r"^Invalid name"):
        format_checkers.check_array_element("not_an_array_element")
