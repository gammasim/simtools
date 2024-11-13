#!/usr/bin/python3

import astropy.units as u
import numpy as np
import pytest

import simtools.utils.value_conversion as value_conversion


def test_extract_type_of_value() -> None:
    """
    Test the extract_type_of_value function.
    """
    # Test with a string.
    assert value_conversion.extract_type_of_value("str") == "str"

    # Test with a list.
    assert value_conversion.extract_type_of_value([1, 2, 3]) == "list"

    # Test with a tuple.
    assert value_conversion.extract_type_of_value((1, 2, 3)) == "tuple"

    # Test with a dictionary
    assert value_conversion.extract_type_of_value({"a": 1, "b": 2}) == "dict"

    # Test with a non-iterable object (int).
    assert value_conversion.extract_type_of_value(123) == "int"

    # Test with a non-iterable object (float).
    assert value_conversion.extract_type_of_value(123.0) == "float"

    # Test with a non-iterable object (bool).
    assert value_conversion.extract_type_of_value(True) == "bool"

    # Test with a non-iterable object (None).
    assert value_conversion.extract_type_of_value(None) == "NoneType"

    # Test with a numpy object (numpy.float64).
    assert value_conversion.extract_type_of_value(np.float64(123.0)) == "float"

    # Test that astropy types are not implemented
    with pytest.raises(NotImplementedError):
        assert value_conversion.extract_type_of_value(1 * u.m)


def test_get_value_unit_type() -> None:
    """Test the get_value_unit_type function."""
    # Test with a string.
    assert value_conversion.get_value_unit_type("hello") == ("hello", None, "str")

    # Test with int.
    assert value_conversion.get_value_unit_type(1) == (1, None, "int")

    # Test with float.
    assert value_conversion.get_value_unit_type(1.0) == (pytest.approx(1.0), None, "float")

    # Test with bool.
    assert value_conversion.get_value_unit_type(True) == (True, None, "bool")

    # Test with None.
    assert value_conversion.get_value_unit_type(None) == (None, None, "NoneType")

    # Test with Quantity.
    assert value_conversion.get_value_unit_type(1 * u.m) == (pytest.approx(1), "m", "float")

    # Test with Quantity.
    assert value_conversion.get_value_unit_type(1.5 * u.cm) == (pytest.approx(1.5), "cm", "float")

    # Test with Quantity with no unit.
    assert value_conversion.get_value_unit_type(1 * u.dimensionless_unscaled) == (
        pytest.approx(1),
        None,
        "float",
    )

    # Test with string representation of Quantity.
    assert value_conversion.get_value_unit_type("1 m") == (pytest.approx(1), "m", "float")

    # test unit fields
    assert value_conversion.get_value_unit_type(1, "m") == (1, "m", "int")
    assert value_conversion.get_value_unit_type(1.0 * u.km, "m") == (1000.0, "m", "float")
    with pytest.raises(u.UnitConversionError):
        value_conversion.get_value_unit_type(1 * u.TeV, "m")

    # cases of simtel-like strings representing arrays
    assert value_conversion.get_value_unit_type("1 2") == ("1 2", None, "str")
    assert value_conversion.get_value_unit_type("0 0") == ("0 0", None, "str")
    assert value_conversion.get_value_unit_type("0. 0. 0.5") == ("0. 0. 0.5", None, "str")


def test_assign_unit_to_quantity():
    assert value_conversion.get_value_as_quantity(10, u.m) == 10 * u.m

    assert value_conversion.get_value_as_quantity(1000 * u.cm, u.m) == 10 * u.m

    with pytest.raises(u.UnitConversionError):
        value_conversion.get_value_as_quantity(1000 * u.TeV, u.m)


def test_split_value_and_unit():
    """Test the split_value_and_unit function."""
    assert value_conversion.split_value_and_unit(100 * u.m) == (100, "m")

    assert value_conversion.split_value_and_unit([100, 200] * u.m) == ([100, 200], ["m", "m"])

    assert value_conversion.split_value_and_unit(np.array([100, 200]) * u.m) == (
        [100, 200],
        ["m", "m"],
    )

    assert value_conversion.split_value_and_unit("100") == (100, None)

    assert value_conversion.split_value_and_unit("100 m") == (100, "m")

    assert value_conversion.split_value_and_unit("100") == (100, None)

    assert value_conversion.split_value_and_unit("hello") == ("hello", None)

    assert value_conversion.split_value_and_unit(["100 m", "200 cm"]) == ([100, 200], ["m", "cm"])

    assert value_conversion.split_value_and_unit(np.array(["100 m", "200 cm"])) == (
        [100, 200],
        ["m", "cm"],
    )

    assert value_conversion.split_value_and_unit([100, "200 cm", 300 * u.m]) == (
        [100, 200, 300],
        [None, "cm", "m"],
    )

    assert value_conversion.split_value_and_unit("100 cm, 200 cm") == ([100, 200], ["cm", "cm"])
