#!/usr/bin/python3

import logging

import astropy.units as u
import numpy as np
import pytest

import simtools.utils.general as gen
import simtools.utils.value_conversion as value_conversion
from simtools.utils.value_conversion import (
    InvalidConfigEntryError,
    MissingRequiredConfigEntryError,
    UnableToIdentifyConfigEntryError,
)


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


def test_validate_config_data(args_dict, io_handler, caplog) -> None:
    parameter_file = io_handler.get_input_data_file(file_name="test_parameters.yml", test=True)
    parameters = gen.collect_data_from_file_or_dict(parameter_file, None)

    validated_data = value_conversion.validate_config_data(
        config_data=None,
        parameters={
            "zenith_angle": {
                "len": 1,
                "default": 20.0,
            }
        },
    )
    assert validated_data.zenith_angle == 20.0

    # Test missing entry
    config_data = {
        "cscat": [0, 10 * u.m, 3 * u.km],
        "source_distance": 20000 * u.m,
        "test_name": 10,
        "dict_par": {"blah": 10, "bleh": 5 * u.m},
    }
    with pytest.raises(MissingRequiredConfigEntryError):
        value_conversion.validate_config_data(config_data=config_data, parameters=parameters)
    assert "Required entry in config_data" in caplog.text

    # Test that a default value is set for a missing parameter.
    config_data["offaxis"] = [0 * u.deg, 0.2 * u.rad, 3 * u.deg]

    validated_data = value_conversion.validate_config_data(
        config_data=config_data | {"azimuth": 0 * u.deg}, parameters=parameters
    )
    assert "zenith_angle" in validated_data._fields
    assert pytest.approx(validated_data.zenith_angle) == 20

    # Test that a None default value is set for a missing parameter.
    config_data["zenith"] = 0 * u.deg
    validated_data = value_conversion.validate_config_data(
        config_data=config_data, parameters=parameters
    )
    assert "azimuth_angle" in validated_data._fields
    assert validated_data.azimuth_angle is None

    # Test a full dictionary
    config_data["azimuth"] = 0 * u.deg

    with caplog.at_level(logging.DEBUG):
        validated_data = value_conversion.validate_config_data(
            config_data=config_data, parameters=parameters
        )
        assert "in config_data cannot be identified" not in caplog.text

    # Testing undefined len
    assert len(validated_data.off_axis_angle) == 3

    # Testing name validation
    assert validated_data.validated_name == 10

    # Testing unit conversion
    assert validated_data.source_distance == 20

    # Testing dict par
    assert validated_data.dict_par["bleh"] == 500

    with caplog.at_level(logging.DEBUG):
        value_conversion.validate_config_data(
            config_data=config_data | {"test": "blah"},
            parameters=parameters,
            ignore_unidentified=True,
        )
        assert "in config_data cannot be identified" in caplog.text

    with pytest.raises(UnableToIdentifyConfigEntryError):
        value_conversion.validate_config_data(
            config_data=config_data | {"test": "blah"}, parameters=parameters
        )


def test_check_value_entry_length() -> None:
    _par_info = {}
    _par_info["len"] = 2
    assert value_conversion._check_value_entry_length([1, 4], "test_1", _par_info) == (2, False)
    _par_info["len"] = None
    assert value_conversion._check_value_entry_length([1, 4], "test_1", _par_info) == (2, True)
    _par_info["len"] = 3
    with pytest.raises(InvalidConfigEntryError):
        value_conversion._check_value_entry_length([1, 4], "test_1", _par_info)
    _par_info.pop("len")
    with pytest.raises(KeyError):
        value_conversion._check_value_entry_length([1, 4], "test_1", _par_info)


def test_validate_and_convert_value_with_units(caplog) -> None:
    _parname = "cscat"
    _parinfo = {
        "len": 5,
        "unit": [None, u.Unit("m"), u.Unit("m"), u.Unit("m"), None],
        "names": ["scat"],
    }
    _value = [0, 10 * u.m, 3 * u.km, "4 m", None]
    _value_keys = ["a", "b", "c", "d", "e"]

    assert value_conversion._validate_and_convert_value_with_units(
        _value, None, _parname, _parinfo
    ) == [
        0,
        10.0,
        3000.0,
        4.0,
        None,
    ]

    assert value_conversion._validate_and_convert_value_with_units(
        _value, _value_keys, _parname, _parinfo
    ) == {
        "a": 0,
        "b": 10.0,
        "c": 3000.0,
        "d": 4.0,
        "e": None,
    }

    _parinfo = {
        "len": None,
        "unit": [None, u.Unit("m"), u.Unit("m"), u.Unit("m"), None],
        "names": ["scat"],
    }
    with pytest.raises(InvalidConfigEntryError):
        value_conversion._validate_and_convert_value_with_units(_value, None, _parname, _parinfo)
    assert "Config entry with undefined length should have a single unit:" in caplog.text
    _parinfo = {
        "len": 5,
        "unit": [None, u.Unit("kg"), u.Unit("m"), u.Unit("m"), None],
        "names": ["scat"],
    }
    with pytest.raises(InvalidConfigEntryError):
        value_conversion._validate_and_convert_value_with_units(_value, None, _parname, _parinfo)
    assert "Config entry given with wrong unit" in caplog.text
    _parinfo = {
        "len": 5,
        "unit": [None, u.Unit("m"), u.Unit("m"), u.Unit("m"), None],
        "names": ["scat"],
    }
    _value = [0, 10 * u.m, 3 * u.km, 4, None]
    with pytest.raises(InvalidConfigEntryError):
        value_conversion._validate_and_convert_value_with_units(_value, None, _parname, _parinfo)
    assert "Config entry given without unit" in caplog.text


def test_validate_and_convert_value_without_units() -> None:
    _parname = "cscat"
    _parinfo = {"len": 3, "names": ["scat"]}
    _value = [0, 10.0, 3.0]
    _value_keys = ["a", "b", "c"]

    assert value_conversion._validate_and_convert_value_without_units(
        _value, None, _parname, _parinfo
    ) == [
        0.0,
        10.0,
        3.0,
    ]
    assert value_conversion._validate_and_convert_value_without_units(
        _value, _value_keys, _parname, _parinfo
    ) == {
        "a": 0,
        "b": 10.0,
        "c": 3.0,
    }
    _value = [0, 10.0 * u.m, 3.0]
    with pytest.raises(InvalidConfigEntryError):
        value_conversion._validate_and_convert_value_without_units(_value, None, _parname, _parinfo)

    assert (
        value_conversion._validate_and_convert_value_without_units(
            ["all"], None, "nightsky_background", {"len": 1}
        )
        == "all"
    )
