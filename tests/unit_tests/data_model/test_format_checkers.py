#!/usr/bin/python3

import pytest

from simtools.data_model import format_checkers


def test_check_astropy_unit():
    assert format_checkers.check_astropy_unit("dimensionless")
    assert format_checkers.check_astropy_unit("m")


def test_check_array_elements():
    assert format_checkers.check_array_element("MSTN-15")

    with pytest.raises(ValueError, match=r"^Invalid name"):
        format_checkers.check_array_element("not_an_array_element")


def test_check_array_trigger_name():
    assert format_checkers.check_array_trigger_name("MSTN_array")
    assert format_checkers.check_array_trigger_name("MSTN_single_telescope")

    with pytest.raises(ValueError, match=r"^Invalid name"):
        format_checkers.check_array_trigger_name("not_an_array_trigger")

    with pytest.raises(ValueError, match=r"^Array trigger name"):
        format_checkers.check_array_trigger_name("MSTN")
