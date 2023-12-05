from copy import copy

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates.errors import UnitsError

import simtools.utils.transformations as transf


def test_rotate_telescope_position(caplog) -> None:
    x = np.array([-10, -10, 10, 10]).astype(float)
    y = np.array([-10.0, 10.0, -10.0, 10.0]).astype(float)
    angle_deg = 30 * u.deg
    x_rot_manual = np.array([-3.7, -13.7, 13.7, 3.7])
    y_rot_manual = np.array([-13.7, 3.7, -3.7, 13.7])

    def check_results(x_to_test, y_to_test, x_right, y_right, angle, theta=0 * u.deg):
        x_rot, y_rot = transf.rotate(x_to_test, y_to_test, angle, theta)
        x_rot, y_rot = np.around(x_rot, 1), np.around(y_rot, 1)
        if not isinstance(x_right, (list, np.ndarray)):
            x_right = [x_right]
        if not isinstance(y_right, (list, np.ndarray)):
            y_right = [y_right]
        for element, _ in enumerate(x_right):
            assert x_right[element] == x_rot[element]
            assert y_right[element] == y_rot[element]

    # Testing without units
    check_results(x, y, x_rot_manual, y_rot_manual, angle_deg)

    # Testing with scalars
    check_results(-10.0, -10.0, -3.7, -13.7, 30 * u.deg)

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
        transf.rotate(x, y[0], angle_deg)
    with pytest.raises(TypeError):
        transf.rotate("1", "2", angle_deg)
        assert "x and y types are not valid! Cannot perform transformation" in caplog.text
    with pytest.raises(TypeError):
        transf.rotate(str(x[0]), y[0], angle_deg)
    with pytest.raises(TypeError):
        transf.rotate(u.Quantity(10), 10, angle_deg)
    with pytest.raises(TypeError):
        transf.rotate(x[0], str(y[0]), angle_deg)
    with pytest.raises(RuntimeError):
        transf.rotate(x[:-1], y, angle_deg)
    with pytest.raises(UnitsError):
        transf.rotate(x_new_array.to(u.cm), y_new_array, angle_deg)
    with pytest.raises(u.core.UnitsError):
        transf.rotate(x_new_array, y_new_array, 30 * u.m)


def test_convert_2D_to_radial_distr(caplog) -> None:
    # Test normal functioning
    max_dist = 100
    bins = 100
    step = max_dist / bins
    xaxis = np.arange(-max_dist, max_dist, step)
    yaxis = np.arange(-max_dist, max_dist, step)
    x2d, y2d = np.meshgrid(xaxis, yaxis)
    distance_to_center_2D = np.sqrt((x2d) ** 2 + (y2d) ** 2)

    distance_to_center_1D, radial_bin_edges = transf.convert_2D_to_radial_distr(
        distance_to_center_2D, xaxis, yaxis, bins=bins, max_dist=max_dist
    )
    difference = radial_bin_edges[:-1] - distance_to_center_1D
    assert pytest.approx(difference[:-1], abs=1) == 0  # last value deviates

    # Test warning in caplog
    transf.convert_2D_to_radial_distr(
        distance_to_center_2D, xaxis, yaxis, bins=4 * bins, max_dist=max_dist
    )
    msg = "The histogram with number of bins"
    assert msg in caplog.text


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
    _no_change_dict_upper = transf.change_dict_keys_case(copy(_upper_dict), False)
    assert _no_change_dict_upper == _upper_dict

    _no_change_dict_lower = transf.change_dict_keys_case(copy(_lower_dict), True)
    assert _no_change_dict_lower == _lower_dict

    _changed_to_lower = transf.change_dict_keys_case(copy(_upper_dict), True)
    assert _changed_to_lower == _lower_dict

    _changed_to_upper = transf.change_dict_keys_case(copy(_lower_dict), False)
    assert _changed_to_upper == _upper_dict

    with pytest.raises(AttributeError):
        transf.change_dict_keys_case([2], False)
        assert "Input is not a proper dictionary" in caplog.text
