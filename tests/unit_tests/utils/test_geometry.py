import logging

import astropy.units as u
import numpy as np
import pytest
from astropy.units import UnitsError

import simtools.utils.geometry as transf


def test_rotate_telescope_position(caplog) -> None:
    x = np.array([-10, -10, 10, 10]).astype(float)
    y = np.array([-10.0, 10.0, -10.0, 10.0]).astype(float)
    angle_deg = 30 * u.deg
    x_rot_manual = np.array([-3.7, -13.7, 13.7, 3.7])
    y_rot_manual = np.array([-13.7, 3.7, -3.7, 13.7])

    def check_results(x_to_test, y_to_test, x_right, y_right, angle, theta=0 * u.deg):
        x_rot, y_rot = transf.rotate(x_to_test, y_to_test, angle, theta)
        x_rot, y_rot = np.around(x_rot, 1), np.around(y_rot, 1)
        if not isinstance(x_right, list | np.ndarray):
            x_right = [x_right]
        if not isinstance(y_right, list | np.ndarray):
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
    with pytest.raises(
        TypeError, match="x and y types are not valid! Cannot perform transformation."
    ):
        transf.rotate("1", "2", angle_deg)
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


def test_convert_2d_to_radial_distr(caplog) -> None:
    # Test normal functioning
    max_dist = 100
    bins = 100
    step = max_dist / bins
    xaxis = np.arange(-max_dist, max_dist, step)
    yaxis = np.arange(-max_dist, max_dist, step)
    x2d, y2d = np.meshgrid(xaxis, yaxis)
    distance_to_center_2d = np.sqrt((x2d) ** 2 + (y2d) ** 2)

    distance_to_center_1d, radial_bin_edges = transf.convert_2d_to_radial_distr(
        distance_to_center_2d, xaxis, yaxis, bins=bins, max_dist=max_dist
    )
    difference = radial_bin_edges[:-1] - distance_to_center_1d
    assert pytest.approx(difference[:-1], abs=1) == 0  # last value deviates

    # Test warning in caplog
    with caplog.at_level(logging.WARNING):
        transf.convert_2d_to_radial_distr(
            distance_to_center_2d, xaxis, yaxis, bins=4 * bins, max_dist=max_dist
        )
    msg = "The histogram with number of bins"
    assert msg in caplog.text


def test_calculate_circular_mean():
    # Test opposite angles cancel out
    angles = np.array([0, np.pi])
    assert transf.calculate_circular_mean(angles) == np.pi / 2

    # Test mean of same angles
    angles = np.array([np.pi / 4, np.pi / 4, np.pi / 4])
    assert transf.calculate_circular_mean(angles) == pytest.approx(np.pi / 4)

    # Test simple cases
    angles = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    assert transf.calculate_circular_mean(angles) == pytest.approx(2.26196)

    # Test mean of random angles
    angles = np.array([0.1, 0.2, 0.3])
    expected = 0.2
    assert transf.calculate_circular_mean(angles) == pytest.approx(expected, abs=1e-6)
