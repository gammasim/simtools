#!/usr/bin/python3

import logging
import time
from copy import copy
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates.errors import UnitsError
from astropy.io.misc import yaml

import simtools.utils.general as gen
from simtools.utils.general import (
    InvalidConfigEntry,
    MissingRequiredConfigEntry,
    UnableToIdentifyConfigEntry,
)

logging.getLogger().setLevel(logging.DEBUG)


def test_collect_dict_data(args_dict, io_handler) -> None:
    in_dict = {"k1": 2, "k2": "bla"}
    dict_for_yaml = {"k3": {"kk3": 4, "kk4": 3.0}, "k4": ["bla", 2]}
    test_yaml_file = io_handler.get_output_file(
        file_name="test_collect_dict_data.yml", dir_type="test"
    )
    if not Path(test_yaml_file).exists():
        with open(test_yaml_file, "w") as output:
            yaml.dump(dict_for_yaml, output, sort_keys=False)

    d1 = gen.collect_data_from_yaml_or_dict(None, in_dict)
    assert "k2" in d1.keys()
    assert d1["k1"] == 2

    d2 = gen.collect_data_from_yaml_or_dict(test_yaml_file, None)
    assert "k3" in d2.keys()
    assert d2["k4"] == ["bla", 2]

    d3 = gen.collect_data_from_yaml_or_dict(test_yaml_file, in_dict)
    assert d3 == d2


def test_collect_dict_from_file() -> None:
    # Test 1: file_path is a yaml file
    file_path = "tests/resources/test_parameters.yml"
    file_name = None
    _dict = gen.collect_dict_from_file(file_path, file_name)
    assert isinstance(_dict, dict)
    assert len(_dict) == 7

    # Test 2: file_path is a directory
    file_path = "tests/resources/"
    file_name = "test_parameters.yml"
    _dict = gen.collect_dict_from_file(file_path, file_name)
    assert isinstance(_dict, dict)
    assert len(_dict) == 7

    # Test 3: file_path is a directory, but file_name is not given
    file_path = "tests/resources/"
    file_name = None
    _dict = gen.collect_dict_from_file(file_path, file_name)
    assert isinstance(_dict, dict)
    assert len(_dict) == 0


def test_validate_config_data(args_dict, io_handler, caplog) -> None:
    parameter_file = io_handler.get_input_data_file(file_name="test_parameters.yml", test=True)
    parameters = gen.collect_data_from_yaml_or_dict(parameter_file, None)

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


def test_validate_and_convert_value_with_units() -> None:
    _parname = "cscat"
    _parinfo = {"len": 4, "unit": [None, u.Unit("m"), u.Unit("m"), None], "names": ["scat"]}
    _value = [0, 10 * u.m, 3 * u.km, None]
    _value_keys = ["a", "b", "c", "d"]

    assert gen._validate_and_convert_value_with_units(_value, None, _parname, _parinfo) == [
        0,
        10.0,
        3000.0,
        None,
    ]

    assert gen._validate_and_convert_value_with_units(_value, _value_keys, _parname, _parinfo) == {
        "a": 0,
        "b": 10.0,
        "c": 3000.0,
        "d": None,
    }

    _parinfo = {"len": None, "unit": [None, u.Unit("m"), u.Unit("m"), None], "names": ["scat"]}
    with pytest.raises(InvalidConfigEntry):
        gen._validate_and_convert_value_with_units(_value, None, _parname, _parinfo)
    _parinfo = {"len": 4, "unit": [None, u.Unit("kg"), u.Unit("m"), None], "names": ["scat"]}
    with pytest.raises(InvalidConfigEntry):
        gen._validate_and_convert_value_with_units(_value, None, _parname, _parinfo)


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


def test_program_is_executable() -> None:
    # (assume 'ls' exist on any system the test is running)
    assert gen.program_is_executable("ls") is not None
    assert gen.program_is_executable("this_program_probably_does_not_exist") is None


def test_change_dict_keys_case() -> None:
    # note that ist entries in DATA_COLUMNS:ATTRIBUTE should not be changed (not keys)
    _upper_dict = {
        "REFERENCE": {"VERSION": "0.1.0"},
        "ACTIVITY": {"NAME": "submit", "ID": "84890304", "DESCRIPTION": "Set data"},
        "DATA_COLUMNS": {"ATTRIBUTE": ["remove_duplicates", "SORT"]},
    }
    _lower_dict = {
        "reference": {"version": "0.1.0"},
        "activity": {"name": "submit", "id": "84890304", "description": "Set data"},
        "data_columns": {"attribute": ["remove_duplicates", "SORT"]},
    }
    _no_change_dict_upper = gen.change_dict_keys_case(copy(_upper_dict), False)
    assert _no_change_dict_upper == _upper_dict

    _no_change_dict_lower = gen.change_dict_keys_case(copy(_lower_dict), True)
    assert _no_change_dict_lower == _lower_dict

    _changed_to_lower = gen.change_dict_keys_case(copy(_upper_dict), True)
    assert _changed_to_lower == _lower_dict

    _changed_to_upper = gen.change_dict_keys_case(copy(_lower_dict), False)
    assert _changed_to_upper == _upper_dict


def test_rotate_telescope_position() -> None:
    x = np.array([-10, -10, 10, 10]).astype(float)
    y = np.array([-10.0, 10.0, -10.0, 10.0]).astype(float)
    angle_deg = 30 * u.deg
    x_rot_manual = np.array([-3.7, -13.7, 13.7, 3.7])
    y_rot_manual = np.array([-13.7, 3.7, -3.7, 13.7])

    def check_results(x_to_test, y_to_test, x_right, y_right, angle, theta=0 * u.deg):
        x_rot, y_rot = gen.rotate(x_to_test, y_to_test, angle, theta)
        x_rot, y_rot = np.around(x_rot, 1), np.around(y_rot, 1)
        for element, _ in enumerate(x):
            assert x_right[element] == x_rot[element]
            assert y_right[element] == y_rot[element]

    # Testing without units
    check_results(x, y, x_rot_manual, y_rot_manual, angle_deg)

    x_new_array, y_new_array = x * u.m, y * u.m
    x_rot_new_array, y_rot_new_array = x_rot_manual * u.m, y_rot_manual * u.m

    # Testing with units
    check_results(x_new_array, y_new_array, x_rot_new_array, y_rot_new_array, angle_deg)

    # Testing with radians
    check_results(x_new_array, y_new_array, x_rot_new_array, y_rot_new_array, angle_deg.to(u.rad))

    # Testing rotation in theta, around Y (3D)
    x_rot_theta_manual = np.array([-2.6, -9.7, 9.7, 2.6])
    y_rot_theta_manual = np.array([-13.7, 3.7, -3.7, 13.7])
    check_results(x, y, x_rot_theta_manual, y_rot_theta_manual, angle_deg, 45 * u.deg)

    with pytest.raises(TypeError):
        gen.rotate(x, y[0], angle_deg)
    with pytest.raises(TypeError):
        gen.rotate(str(x[0]), y[0], angle_deg)
    with pytest.raises(TypeError):
        gen.rotate(u.Quantity(10), 10, angle_deg)
    with pytest.raises(TypeError):
        gen.rotate(x[0], str(y[0]), angle_deg)
    with pytest.raises(RuntimeError):
        gen.rotate(x[:-1], y, angle_deg)
    with pytest.raises(UnitsError):
        gen.rotate(x_new_array.to(u.cm), y_new_array, angle_deg)
    with pytest.raises(u.core.UnitsError):
        gen.rotate(x_new_array, y_new_array, 30 * u.m)


def test_convert_2D_to_radial_distr(caplog) -> None:
    # Test normal functioning
    max_dist = 100
    bins = 100
    step = max_dist / bins
    xaxis = np.arange(-max_dist, max_dist, step)
    yaxis = np.arange(-max_dist, max_dist, step)
    x2d, y2d = np.meshgrid(xaxis, yaxis)
    distance_to_center_2D = np.sqrt((x2d) ** 2 + (y2d) ** 2)

    distance_to_center_1D, radial_edges = gen.convert_2D_to_radial_distr(
        distance_to_center_2D, xaxis, yaxis, bins=bins, max_dist=max_dist
    )
    difference = radial_edges[:-1] - distance_to_center_1D
    assert pytest.approx(difference[:-1], abs=1) == 0  # last value deviates

    # Test warning in caplog
    gen.convert_2D_to_radial_distr(
        distance_to_center_2D, xaxis, yaxis, bins=4 * bins, max_dist=max_dist
    )
    msg = "The histogram with number of bins"
    assert msg in caplog.text


def test_save_dict_to_file(tmp_test_directory, caplog) -> None:
    paths = ["test_file", "test_file.yml"]
    example_dict = {"key": 12}
    for path in paths:
        gen.save_dict_to_file(example_dict, f"{tmp_test_directory}/{path}")
        with open(f"{tmp_test_directory}/test_file.yml", encoding="utf-8") as file:
            new_example_dict = yaml.load(file)
            assert new_example_dict == example_dict

    # Path
    path = tmp_test_directory / "test_file_2.yml"
    example_dict = {"key": 12}
    gen.save_dict_to_file(example_dict, path)
    with open(path, encoding="utf-8") as file:
        new_example_dict = yaml.load(file)
        assert new_example_dict == example_dict

    # Test error
    path = tmp_test_directory / "non_existing_path/test_file_2.yml"
    example_dict = {"key": 12}
    with pytest.raises(IOError):
        gen.save_dict_to_file(example_dict, path)
        assert "Failed to write to" in caplog.text


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
