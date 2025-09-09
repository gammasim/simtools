import logging

import astropy.units as u
import numpy as np
import pytest
from astropy.tests.helper import assert_quantity_allclose
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
    assert transf.calculate_circular_mean(angles) == pytest.approx(0.2, abs=1e-6)


def test_solid_angle():
    # Test with angle in radians
    angle_rad = 1 * u.rad
    expected_solid_angle_rad = 2 * np.pi * (1 - np.cos(angle_rad)) * u.sr
    assert_quantity_allclose(transf.solid_angle(angle_rad), expected_solid_angle_rad)

    # Test with angle in degrees
    angle_deg = 90 * u.deg
    expected_solid_angle_deg = 2 * np.pi * (1 - np.cos(angle_deg.to(u.rad))) * u.sr
    assert_quantity_allclose(transf.solid_angle(angle_deg), expected_solid_angle_deg)

    # Test with zero angle
    angle_zero = 0 * u.rad
    assert_quantity_allclose(transf.solid_angle(angle_zero), 0 * u.sr)

    # Test with a full circle (360 degrees)
    angle_full_circle = 360 * u.deg
    expected_solid_angle_full_circle = 2 * np.pi * (1 - np.cos(angle_full_circle.to(u.rad))) * u.sr
    assert_quantity_allclose(
        transf.solid_angle(angle_full_circle), expected_solid_angle_full_circle
    )


def test_transform_ground_to_shower_coordinates():
    """
    Test ground to shower coordinates.

    Values below crosschecked with Eventdisplay and ctapipe results.

    For ctapipe, do:

        from ctapipe.coordinates import GroundFrame, TiltedGroundFrame

        ground = GroundFrame(x=x_core * u.m, y=y_core * u.m, z=np.zeros_like(x_core) * u.m)
        shower_frame = ground.transform_to(
            TiltedGroundFrame(
                pointing_direction=AltAz(
                    az=shower_azimuth * u.rad, alt=shower_altitude * u.rad
                )
            )
        )
        return shower_frame.x.value, shower_frame.y.value
    """
    x_ground = np.array([488.83758545] * 4)
    y_ground = np.array([-901.18658447] * 4)
    z_ground = np.array([0.0] * 4)

    # Following cases are tested:
    # 1. both systems are identical for zenith pointing and zero azimuth
    # 2. zenith pointing with azimuth rotation by 90 deg
    # 3. random values
    # 4. pointing towards horizon
    shower_azimuth = np.array([0.0, np.pi / 2.0, 0.21440187, 0.0])
    shower_altitude = np.array([np.pi / 2.0, np.pi / 2.0, 1.29735112, 0.0])

    # The following expected values were crosschecked with Eventdisplay and ctapipe.
    # For reference, see the docstring above for the ctapipe code used.
    # The third and fourth columns are the results of transforming the ground coordinates
    # (x_ground, y_ground, z_ground) to the shower frame for the given azimuth and altitude.
    # These values are hardcoded here for regression testing.
    expected_x = np.array(
        [
            x_ground[0],  # Case 1: zenith pointing, zero azimuth
            -1.0 * y_ground[0],  # Case 2: zenith pointing, azimuth 90 deg
            651.6379522993169,  # Case 3: expected result from Eventdisplay/ctapipe
            0.0,  # Case 4: pointing towards horizon
        ]
    )
    expected_y = np.array(
        [
            y_ground[0],  # Case 1
            x_ground[0],  # Case 2
            -780.4105314700417,  # Case 3 expected result from Eventdisplay/ctapipe
            y_ground[0],  # Case 4
        ]
    )
    # The following values were obtained by running the Eventdisplay code.
    # Cross-checked with the ctapipe code in the docstring.
    expected_z = np.array(
        [
            0.0,  # Case 1
            0.0,  # Case 2
            -132.01070589573638,  # Case 3: expected result from Eventdisplay/ctapipe
            -1.0 * x_ground[0],  # Case 4: for horizon, z = -x_ground
        ]
    )
    expected = np.array([expected_x, expected_y, expected_z])

    result = transf.transform_ground_to_shower_coordinates(
        x_ground, y_ground, z_ground, shower_azimuth, shower_altitude
    )

    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1.0e-10)


def test_fiducial_radius_from_shape():
    # Test for circle (shape = 0)
    width_circle = 10.0
    shape_circle = 0
    expected_radius_circle = 5.0
    assert transf.fiducial_radius_from_shape(width_circle, shape_circle) == pytest.approx(
        expected_radius_circle
    )

    # Test for square (shape = 2)
    width_square = 10.0
    shape_square = 2
    expected_radius_square = 10.0 / np.sqrt(2.0)
    assert transf.fiducial_radius_from_shape(width_square, shape_square) == pytest.approx(
        expected_radius_square
    )

    # Test for hexagon (shape = 1)
    width_hexagon_1 = 10.0
    shape_hexagon_1 = 1
    expected_radius_hexagon_1 = 10.0 / np.sqrt(3.0)
    assert transf.fiducial_radius_from_shape(width_hexagon_1, shape_hexagon_1) == pytest.approx(
        expected_radius_hexagon_1
    )

    # Test for hexagon (shape = 3)
    width_hexagon_3 = 10.0
    shape_hexagon_3 = 3
    expected_radius_hexagon_3 = 10.0 / np.sqrt(3.0)
    assert transf.fiducial_radius_from_shape(width_hexagon_3, shape_hexagon_3) == pytest.approx(
        expected_radius_hexagon_3
    )

    # Test for invalid shape
    with pytest.raises(ValueError, match="Unknown shape value 4"):
        transf.fiducial_radius_from_shape(10.0, 4)
