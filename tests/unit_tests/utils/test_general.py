#!/usr/bin/python3

import gzip
import logging
import os
import time
import urllib.error
from copy import copy
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.io.misc import yaml

import simtools.utils.general as gen
from simtools.utils.general import (
    InvalidConfigData,
    InvalidConfigEntry,
    MissingRequiredConfigEntry,
    UnableToIdentifyConfigEntry,
)

logging.getLogger().setLevel(logging.DEBUG)


def test_collect_dict_data(args_dict, io_handler, caplog) -> None:
    in_dict = {"k1": 2, "k2": "bla"}
    dict_for_yaml = {"k3": {"kk3": 4, "kk4": 3.0}, "k4": ["bla", 2]}
    test_yaml_file = io_handler.get_output_file(
        file_name="test_collect_dict_data.yml", dir_type="test"
    )
    if not Path(test_yaml_file).exists():
        with open(test_yaml_file, "w") as output:
            yaml.dump(dict_for_yaml, output, sort_keys=False)

    d1 = gen.collect_data_from_file_or_dict(None, in_dict)
    assert "k2" in d1.keys()
    assert d1["k1"] == 2

    d2 = gen.collect_data_from_file_or_dict(test_yaml_file, None)
    assert "k3" in d2.keys()
    assert d2["k4"] == ["bla", 2]

    d3 = gen.collect_data_from_file_or_dict(test_yaml_file, in_dict)
    assert d3 == d2

    assert gen.collect_data_from_file_or_dict(None, None, allow_empty=True) is None
    assert "Input has not been provided (neither by file, nor by dict)" in caplog.text

    with pytest.raises(InvalidConfigData):
        gen.collect_data_from_file_or_dict(None, None, allow_empty=False)
        assert "Input has not been provided (neither by file, nor by dict)" in caplog.text


def test_collect_dict_from_url(io_handler) -> None:
    _file = "tests/resources/test_parameters.yml"
    _reference_dict = gen.collect_data_from_file_or_dict(_file, None)

    _url = "https://raw.githubusercontent.com/gammasim/simtools/main/"
    _url_dict = gen.collect_data_from_http(_url + _file)

    assert _reference_dict == _url_dict

    _dict = gen.collect_data_from_file_or_dict(_url + _file, None)
    assert isinstance(_dict, dict)
    assert len(_dict) > 0

    _url = "https://raw.githubusercontent.com/gammasim/simtools/not_main/"
    with pytest.raises(urllib.error.HTTPError):
        gen.collect_data_from_http(_url + _file)


def test_validate_config_data(args_dict, io_handler, caplog) -> None:
    parameter_file = io_handler.get_input_data_file(file_name="test_parameters.yml", test=True)
    parameters = gen.collect_data_from_file_or_dict(parameter_file, None)

    validated_data = gen.validate_config_data(
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
    with pytest.raises(MissingRequiredConfigEntry):
        validated_data = gen.validate_config_data(config_data=config_data, parameters=parameters)
        assert "Required entry in config_data" in caplog.text

    # Test that a default value is set for a missing parameter.
    config_data["offaxis"] = [0 * u.deg, 0.2 * u.rad, 3 * u.deg]

    validated_data = gen.validate_config_data(
        config_data=config_data | {"azimuth": 0 * u.deg}, parameters=parameters
    )
    assert "zenith_angle" in validated_data._fields
    assert pytest.approx(validated_data.zenith_angle) == 20

    # Test that a None default value is set for a missing parameter.
    config_data["zenith"] = 0 * u.deg
    validated_data = gen.validate_config_data(config_data=config_data, parameters=parameters)
    assert "azimuth_angle" in validated_data._fields
    assert validated_data.azimuth_angle is None

    # Test a full dictionary
    config_data["azimuth"] = 0 * u.deg

    with caplog.at_level(logging.DEBUG):
        validated_data = gen.validate_config_data(config_data=config_data, parameters=parameters)
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
        gen.validate_config_data(
            config_data=config_data | {"test": "blah"},
            parameters=parameters,
            ignore_unidentified=True,
        )
        assert "in config_data cannot be identified" in caplog.text

    with pytest.raises(UnableToIdentifyConfigEntry):
        gen.validate_config_data(config_data=config_data | {"test": "blah"}, parameters=parameters)


def test_check_value_entry_length() -> None:
    _par_info = {}
    _par_info["len"] = 2
    assert gen._check_value_entry_length([1, 4], "test_1", _par_info) == (2, False)
    _par_info["len"] = None
    assert gen._check_value_entry_length([1, 4], "test_1", _par_info) == (2, True)
    _par_info["len"] = 3
    with pytest.raises(InvalidConfigEntry):
        gen._check_value_entry_length([1, 4], "test_1", _par_info)
    _par_info.pop("len")
    with pytest.raises(KeyError):
        gen._check_value_entry_length([1, 4], "test_1", _par_info)


def test_validate_and_convert_value_with_units(caplog) -> None:
    _parname = "cscat"
    _parinfo = {
        "len": 5,
        "unit": [None, u.Unit("m"), u.Unit("m"), u.Unit("m"), None],
        "names": ["scat"],
    }
    _value = [0, 10 * u.m, 3 * u.km, "4 m", None]
    _value_keys = ["a", "b", "c", "d", "e"]

    assert gen._validate_and_convert_value_with_units(_value, None, _parname, _parinfo) == [
        0,
        10.0,
        3000.0,
        4.0,
        None,
    ]

    assert gen._validate_and_convert_value_with_units(_value, _value_keys, _parname, _parinfo) == {
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
    with pytest.raises(InvalidConfigEntry):
        gen._validate_and_convert_value_with_units(_value, None, _parname, _parinfo)
        assert "Config entry given with wrong unit" in caplog.text
    _parinfo = {
        "len": 5,
        "unit": [None, u.Unit("kg"), u.Unit("m"), u.Unit("m"), None],
        "names": ["scat"],
    }
    with pytest.raises(InvalidConfigEntry):
        gen._validate_and_convert_value_with_units(_value, None, _parname, _parinfo)
        assert "Config entry given with wrong unit" in caplog.text
    _parinfo = {
        "len": 5,
        "unit": [None, u.Unit("m"), u.Unit("m"), u.Unit("m"), None],
        "names": ["scat"],
    }
    _value = [0, 10 * u.m, 3 * u.km, 4, None]
    with pytest.raises(InvalidConfigEntry):
        gen._validate_and_convert_value_with_units(_value, None, _parname, _parinfo)
        assert "Config entry given without unit" in caplog.text


def test_validate_and_convert_value_without_units() -> None:
    _parname = "cscat"
    _parinfo = {"len": 3, "names": ["scat"]}
    _value = [0, 10.0, 3.0]
    _value_keys = ["a", "b", "c"]

    assert gen._validate_and_convert_value_without_units(_value, None, _parname, _parinfo) == [
        0.0,
        10.0,
        3.0,
    ]
    assert gen._validate_and_convert_value_without_units(
        _value, _value_keys, _parname, _parinfo
    ) == {
        "a": 0,
        "b": 10.0,
        "c": 3.0,
    }
    _value = [0, 10.0 * u.m, 3.0]
    with pytest.raises(InvalidConfigEntry):
        gen._validate_and_convert_value_without_units(_value, None, _parname, _parinfo)

    assert (
        gen._validate_and_convert_value_without_units(
            ["all"], None, "nightsky_background", {"len": 1}
        )
        == "all"
    )


def test_program_is_executable(caplog) -> None:
    # (assume 'ls' exist on any system the test is running)
    assert gen.program_is_executable("ls") is not None
    assert gen.program_is_executable("/bin/ls") is not None  # The actual path should not matter
    assert gen.program_is_executable("this_program_probably_does_not_exist") is None
    os.environ.pop("PATH", None)
    assert gen.program_is_executable("this_program_probably_does_not_exist") is None
    assert "PATH environment variable is not set." in caplog.text


def test_get_file_age(tmp_test_directory) -> None:
    # Create a temporary file and wait for 1 seconds before accessing it
    with open(tmp_test_directory / "test_file.txt", "w", encoding="utf-8") as file:
        file.write("Test data")

    time.sleep(1)

    try:
        age_in_minutes = gen.get_file_age(tmp_test_directory / "test_file.txt")
        # Age should be within an acceptable range (0 to 0.05 minutes or 3 seconds)
        assert 0 <= age_in_minutes <= 0.05
    except FileNotFoundError:
        pytest.fail("get_file_age raised FileNotFoundError for an existing file.")

    # Ensure that the function raises FileNotFoundError for a non-existent file
    with pytest.raises(FileNotFoundError):
        gen.get_file_age(tmp_test_directory / "nonexistent_file.txt")


def test_separate_args_and_config_data() -> None:
    # Test the function "separate_args_and_config_data"
    expected_args = ["arg1", "arg2"]
    kwargs = {"arg1": 1, "arg2": 2, "arg3": 3}
    args, config_data = gen.separate_args_and_config_data(expected_args, **kwargs)
    assert args == {"arg1": 1, "arg2": 2}
    assert config_data == {"arg3": 3}


def test_get_log_excerpt(tmp_test_directory) -> None:
    log_file = tmp_test_directory / "log.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("This is a log file.\n")
        f.write("This is the second line of the log file.\n")

    assert gen.get_log_excerpt(log_file) == (
        "\n\nRuntime error - See below the relevant part of the log/err file.\n\n"
        f"{log_file}\n"
        "====================================================================\n\n"
        "This is a log file."
        "This is the second line of the log file.\n\n"
        "====================================================================\n"
    )


def test_file_has_text(tmp_test_directory, caplog, file_has_text) -> None:
    """Test the file_has_text function."""

    # Test with file that has text.
    file = tmp_test_directory / "test_file_has_text.txt"
    text = "test"
    with open(file, "w") as f:
        f.write(text)
    assert file_has_text(file, text)
    assert not file_has_text(file, "test2")

    # Test with empty file.
    file = tmp_test_directory / "test_file_is_empty.txt"
    with open(file, "w") as f:
        f.write("")
    assert not file_has_text(file, text)

    # Test with file that does not exist.
    file = tmp_test_directory / "test_file_does_not_exist.txt"
    assert not file_has_text(file, text)


def test_collect_kwargs() -> None:
    """
    Test the collect_kwargs function.
    """

    # Test with no kwargs.
    kwargs = {}
    out_kwargs = gen.collect_kwargs("label", kwargs)
    assert out_kwargs == {}

    # Test with one kwargs.
    kwargs = {"label_a": 1}
    out_kwargs = gen.collect_kwargs("label", kwargs)
    assert out_kwargs == {"a": 1}

    # Test with multiple kwargs.
    kwargs = {"label_a": 1, "label_b": 2, "label_c": 3}
    out_kwargs = gen.collect_kwargs("label", kwargs)
    assert out_kwargs == {"a": 1, "b": 2, "c": 3}

    # Test with kwargs where only one starts with label_.
    kwargs = {"a": 1, "b": 2, "label_c": 3, "d": 4}
    out_kwargs = gen.collect_kwargs("label", kwargs)
    assert out_kwargs == {"c": 3}

    # Test with kwargs that do not start with label_.
    kwargs = {"a": 1, "b": 2, "c": 3, "d": 4}
    out_kwargs = gen.collect_kwargs("label", kwargs)
    assert out_kwargs == {}


def test_set_default_kwargs() -> None:
    """
    Test the set_default_kwargs function.
    """

    in_kwargs = {"a": 1, "b": 2}
    out_kwargs = gen.set_default_kwargs(in_kwargs, c=3, d=4)
    assert out_kwargs == {"a": 1, "b": 2, "c": 3, "d": 4}


def test_collect_final_lines(tmp_test_directory) -> None:
    """
    Test the collect_final_lines function.
    """

    # Test with no file.
    with pytest.raises(FileNotFoundError):
        gen.collect_final_lines("no_such_file.txt", 10)

    # Test with empty file.
    file = tmp_test_directory / "empty_file.txt"
    with open(file, "w"):
        pass
    assert gen.collect_final_lines(file, 10) == ""

    # Test with one line file.
    file = tmp_test_directory / "one_line_file.txt"
    with open(file, "w") as f:
        f.write("Line 1")
    assert gen.collect_final_lines(file, 1) == "Line 1"

    # In the following tests the \n in the output are removed, but in the actual print statements,
    # where the original function is used, they are still present in the string representation.

    # Test with multiple lines file.
    file = tmp_test_directory / "multiple_lines_file.txt"
    with open(file, "w") as f:
        f.write("Line 1\nLine 2\nLine 3")
    assert gen.collect_final_lines(file, 2) == "Line 2Line 3"

    # Test with file with n_lines lines.
    file = tmp_test_directory / "n_lines_file.txt"
    with open(file, "w") as f:
        for i in range(10):
            f.write(f"Line {i}\n")
        f.write("Line 10")
    assert gen.collect_final_lines(file, 3) == "Line 8Line 9Line 10"

    # Test with file compressed in gzip.
    file = tmp_test_directory / "compressed_file.txt.gz"
    with gzip.open(file, "wb") as f:
        f.write(b"Line 1\nLine 2\nLine 3")
    assert gen.collect_final_lines(file, 2) == "Line 2Line 3"


def test_log_level_from_user() -> None:
    """
    Test get_log_level_from_user() function.
    """
    assert gen.get_log_level_from_user("info") == logging.INFO
    assert gen.get_log_level_from_user("debug") == logging.DEBUG
    assert gen.get_log_level_from_user("warn") == logging.WARNING
    assert gen.get_log_level_from_user("warning") == logging.WARNING
    assert gen.get_log_level_from_user("error") == logging.ERROR
    assert gen.get_log_level_from_user("critical") == logging.CRITICAL

    with pytest.raises(ValueError):
        gen.get_log_level_from_user("invalid")
    with pytest.raises(AttributeError):
        gen.get_log_level_from_user(1)
    with pytest.raises(AttributeError):
        gen.get_log_level_from_user(None)
    with pytest.raises(AttributeError):
        gen.get_log_level_from_user(True)


def test_copy_as_list() -> None:
    """
    Test the copy_as_list function.
    """

    # Test with a string.
    assert gen.copy_as_list("str") == ["str"]

    # Test with a list.
    assert gen.copy_as_list([1, 2, 3]) == [1, 2, 3]

    # Test with a tuple.
    assert gen.copy_as_list((1, 2, 3)) == [1, 2, 3]

    # Test with a dictionary (probably not really a useful case, but should test anyway).
    assert gen.copy_as_list({"a": 1, "b": 2}) == ["a", "b"]

    # Test with a non-iterable object.
    assert gen.copy_as_list(123) == [123]


def test_find_file_in_current_directory(tmp_test_directory) -> None:
    """
    Test finding a file in the temp test directory directory.
    """
    file_name = tmp_test_directory / "test.txt"
    with open(file_name, "w") as file:
        file.write("Test data")
    file_path = gen.find_file(file_name, tmp_test_directory)
    assert file_path == file_name


def test_find_file_in_non_existing_directory(tmp_test_directory) -> None:
    """
    Test finding a file in a non-existing directory.
    """
    file_name = tmp_test_directory / "test.txt"

    loc = Path("non_existing_directory")
    with pytest.raises(FileNotFoundError):
        gen.find_file(file_name, loc)


def test_find_file_recursively(tmp_test_directory) -> None:
    """
    Test finding a file recursively.
    """
    file_name = "test_1.txt"
    test_directory_sub_dir = tmp_test_directory / "test"
    Path(test_directory_sub_dir).mkdir(parents=True, exist_ok=True)
    with open(test_directory_sub_dir / file_name, "w", encoding="utf-8") as file:
        file.write("Test data")
    loc = tmp_test_directory
    file_path = gen.find_file(file_name, loc)
    assert file_path == Path(loc).joinpath("test").joinpath(file_name)

    # Test also the case in which we recursively find unrelated files.
    file_name = "test_2.txt"
    Path(test_directory_sub_dir / "unrelated_sub_dir").mkdir(parents=True, exist_ok=True)
    with open(
        test_directory_sub_dir / "unrelated_sub_dir" / "unrelated_file.txt", "w", encoding="utf-8"
    ) as file:
        file.write("Test data")
    loc = tmp_test_directory
    with pytest.raises(FileNotFoundError):
        gen.find_file(file_name, loc)


def test_find_file_not_found(tmp_test_directory) -> None:
    """
    Test finding a file that does not exist.
    """
    file_name = "not_existing_file.txt"
    loc = Path(tmp_test_directory)
    with pytest.raises(FileNotFoundError):
        gen.find_file(file_name, loc)


def test_is_url():
    url = "http://www.desy.de"
    assert gen.is_url(url) is True

    url = "ftp://www.desy.de"
    assert gen.is_url(url) is True

    url = ""
    assert gen.is_url(url) is False

    url = "http://"
    assert gen.is_url(url) is False

    url = "desy.de"
    assert gen.is_url(url) is False

    assert gen.is_url(5.0) is False


def test_collect_data_dict_from_json():
    file = "tests/resources/reference_point_altitude.json"
    data = gen.collect_data_from_file_or_dict(file, None)
    assert len(data) == 5
    assert data["units"] == "m"


def test_collect_data_from_http():
    file = "tests/resources/test_parameters.yml"
    url = "https://raw.githubusercontent.com/gammasim/simtools/main/"

    data = gen.collect_data_from_http(url + file)
    assert isinstance(data, dict)

    file = "tests/resources/reference_point_altitude.json"
    data = gen.collect_data_from_http(url + file)
    assert isinstance(data, dict)

    file = "tests/resources/simtel_histograms_file_list.txt"
    with pytest.raises(TypeError):
        data = gen.collect_data_from_http(url + file)

    url = "https://raw.githubusercontent.com/gammasim/simtools/not_right/"
    with pytest.raises(urllib.error.HTTPError):
        data = gen.collect_data_from_http(url + file)


def test_change_dict_keys_case(caplog) -> None:
    # note that ist entries in DATA_COLUMNS:ATTRIBUTE should not be changed (not keys)
    _upper_dict = {
        "REFERENCE": {"VERSION": "0.1.0"},
        "ACTIVITY": {"NAME": "submit", "ID": "84890304", "DESCRIPTION": "Set data"},
        "DATA_COLUMNS": {"ATTRIBUTE": ["remove_duplicates", "SORT"]},
        "DICT_IN_LIST": {
            "KEY_OF_FIRST_DICT": ["FIRST_ITEM", {"KEY_OF_NESTED_DICT": "VALUE_OF_SECOND_DICT"}]
        },
    }
    _lower_dict = {
        "reference": {"version": "0.1.0"},
        "activity": {"name": "submit", "id": "84890304", "description": "Set data"},
        "data_columns": {"attribute": ["remove_duplicates", "SORT"]},
        "dict_in_list": {
            "key_of_first_dict": ["FIRST_ITEM", {"key_of_nested_dict": "VALUE_OF_SECOND_DICT"}]
        },
    }
    _no_change_dict_upper = gen.change_dict_keys_case(copy(_upper_dict), False)
    assert _no_change_dict_upper == _upper_dict

    _no_change_dict_lower = gen.change_dict_keys_case(copy(_lower_dict), True)
    assert _no_change_dict_lower == _lower_dict

    _changed_to_lower = gen.change_dict_keys_case(copy(_upper_dict), True)
    assert _changed_to_lower == _lower_dict

    _changed_to_upper = gen.change_dict_keys_case(copy(_lower_dict), False)
    assert _changed_to_upper == _upper_dict

    with pytest.raises(AttributeError):
        gen.change_dict_keys_case([2], False)
        assert "Input is not a proper dictionary" in caplog.text


def test_sort_arrays() -> None:
    """
    Test the sort_arrays function.
    """

    # Test with no arguments.
    args = []
    new_args = gen.sort_arrays(*args)
    assert not new_args

    # Test with one argument.
    args = [list(range(10))]
    new_args = gen.sort_arrays(*args)
    assert new_args == [list(range(10))]

    # Test with multiple arguments.
    args = [list(range(10)), list(range(10, 20))]
    new_args = gen.sort_arrays(*args)
    assert new_args == [list(range(10)), list(range(10, 20))]

    # Test with arguments of different lengths.
    args = [list(range(10)), list(range(5))]
    new_args = gen.sort_arrays(*args)
    assert new_args == [list(range(10)), list(range(5))]

    # Test with arguments that are not arrays.
    args = [1, 2, 3]
    with pytest.raises(TypeError):
        gen.sort_arrays(*args)

    # Test with the input array not in the right order.
    args = [list(reversed(range(10)))]
    new_args = gen.sort_arrays(*args)
    assert new_args == [list(range(10))]


@pytest.mark.parametrize(
    "input_data, expected_output",
    [
        (
            {
                "key1": "This is a string\n with a newline",
                "key2": ["List item 1\n", "List item 2\n"],
                "key3": {"nested_key": "Nested string\n with a newline"},
                "key4": [{"nested_dict": "string2\n"}, {"nested_dict2": "string3\n"}],
            },
            {
                "key1": "This is a string with a newline",
                "key2": ["List item 1", "List item 2"],
                "key3": {"nested_key": "Nested string with a newline"},
                "key4": [{"nested_dict": "string2"}, {"nested_dict2": "string3"}],
            },
        ),
    ],
)
def test_remove_substring_recursively_from_dict(input_data, expected_output, caplog):
    result = gen.remove_substring_recursively_from_dict(input_data, "\n")
    assert result == expected_output

    # no error should be raised for None input, but a debug message should be printed
    gen._logger.setLevel(logging.DEBUG)
    gen.remove_substring_recursively_from_dict([2])
    assert any(
        record.levelname == "DEBUG" and "Input is not a dictionary: [2]" in record.message
        for record in caplog.records
    )


def test_extract_type_of_value() -> None:
    """
    Test the extract_type_of_value function.
    """
    # Test with a string.
    assert gen.extract_type_of_value("str") == "str"

    # Test with a list.
    assert gen.extract_type_of_value([1, 2, 3]) == "list"

    # Test with a tuple.
    assert gen.extract_type_of_value((1, 2, 3)) == "tuple"

    # Test with a dictionary
    assert gen.extract_type_of_value({"a": 1, "b": 2}) == "dict"

    # Test with a non-iterable object (int).
    assert gen.extract_type_of_value(123) == "int"

    # Test with a non-iterable object (float).
    assert gen.extract_type_of_value(123.0) == "float"

    # Test with a non-iterable object (bool).
    assert gen.extract_type_of_value(True) == "bool"

    # Test with a non-iterable object (None).
    assert gen.extract_type_of_value(None) == "NoneType"

    # Test with a numpy object (numpy.float64).
    assert gen.extract_type_of_value(np.float64(123.0)) == "float"

    # Test that astropy types are not implemented
    with pytest.raises(NotImplementedError):
        assert gen.extract_type_of_value(1 * u.m)


def test_get_value_unit_type() -> None:
    """Test the get_value_unit_type function."""
    # Test with a string.
    assert gen.get_value_unit_type("hello") == ("hello", None, "str")

    # Test with int.
    assert gen.get_value_unit_type(1) == (1, None, "int")

    # Test with float.
    assert gen.get_value_unit_type(1.0) == (pytest.approx(1.0), None, "float")

    # Test with bool.
    assert gen.get_value_unit_type(True) == (True, None, "bool")

    # Test with None.
    assert gen.get_value_unit_type(None) == (None, None, "NoneType")

    # Test with Quantity.
    assert gen.get_value_unit_type(1 * u.m) == (pytest.approx(1), "m", "float")

    # Test with Quantity.
    assert gen.get_value_unit_type(1.5 * u.cm) == (pytest.approx(1.5), "cm", "float")

    # Test with Quantity with no unit.
    assert gen.get_value_unit_type(1 * u.dimensionless_unscaled) == (
        pytest.approx(1),
        None,
        "float",
    )

    # Test with string representation of Quantity.
    assert gen.get_value_unit_type("1 m") == (pytest.approx(1), "m", "float")
