#!/usr/bin/python3

import logging
from copy import copy
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates.errors import UnitsError
from astropy.io.misc import yaml

import simtools.util.general as gen
from simtools.util.general import InvalidConfigEntry

logging.getLogger().setLevel(logging.DEBUG)


def test_collect_dict_data(args_dict, io_handler):
    in_dict = {"k1": 2, "k2": "bla"}
    dict_for_yaml = {"k3": {"kk3": 4, "kk4": 3.0}, "k4": ["bla", 2]}
    test_yaml_file = io_handler.get_output_file(file_name="test_collect_dict_data.yml", test=True)
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


def test_validate_config_data(args_dict, io_handler):

    parameter_file = io_handler.get_input_data_file(file_name="test_parameters.yml", test=True)
    parameters = gen.collect_data_from_yaml_or_dict(parameter_file, None)

    config_data = {
        "zenith": 0 * u.deg,
        "offaxis": [0 * u.deg, 0.2 * u.rad, 3 * u.deg],
        "cscat": [0, 10 * u.m, 3 * u.km],
        "source_distance": 20000 * u.m,
        "test_name": 10,
        "dict_par": {"blah": 10, "bleh": 5 * u.m},
    }

    validated_data = gen.validate_config_data(config_data=config_data, parameters=parameters)

    # Testing undefined len
    assert len(validated_data.off_axis_angle) == 3

    # Testing name validation
    assert validated_data.validated_name == 10

    # Testing unit conversion
    assert validated_data.source_distance == 20

    # Testing dict par
    assert validated_data.dict_par["bleh"] == 500


def test_check_value_entry_length():

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


def test_validate_and_convert_value_with_units():

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


def test_validate_and_convert_value_without_units():

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


def test_program_is_executable():

    # (assume 'ls' exist on any system the test is running)
    assert gen.program_is_executable("ls") is not None
    assert gen.program_is_executable("this_program_probably_does_not_exist") is None


def test_change_dict_keys_case():

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


def test_rotate_telescope_position():
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


def test_convert_2D_to_radial_distr(caplog):

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


def test_save_dict_to_file(tmp_test_directory):

    # str
    paths = ["test_file", "test_file.yml"]
    example_dict = {"key": 12}
    for path in paths:
        gen.save_dict_to_file(example_dict, f"{tmp_test_directory}/{path}")
        with open(f"{tmp_test_directory}/test_file.yml") as file:
            new_example_dict = yaml.load(file)
            assert new_example_dict == example_dict

    # Path
    path = tmp_test_directory / "test_file_2.yml"
    example_dict = {"key": 12}
    gen.save_dict_to_file(example_dict, path)
    with open(path) as file:
        new_example_dict = yaml.load(file)
        assert new_example_dict == example_dict
