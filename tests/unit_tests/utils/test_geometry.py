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
        TypeError, match=r"x and y types are not valid! Cannot perform transformation."
    ):
        transf.rotate("1", "2", angle_deg, 0 * u.deg)
    with pytest.raises(TypeError):
        transf.rotate(str(x[0]), y[0], angle_deg, 0 * u.deg)
    with pytest.raises(TypeError):
        transf.rotate(u.Quantity(10), 10, angle_deg, 0 * u.deg)
    with pytest.raises(TypeError):
        transf.rotate(x[0], str(y[0]), angle_deg, 0 * u.deg)
    with pytest.raises(RuntimeError):
        transf.rotate(x[:-1], y, angle_deg)
    with pytest.raises(UnitsError):
        transf.rotate(x_new_array.to(u.cm), y_new_array, angle_deg)
    with pytest.raises(u.core.UnitsError):
        transf.rotate(x_new_array, y_new_array, 30 * u.m)


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
    with pytest.raises(ValueError, match=r"Unknown shape value 4\. Valid values are:"):
        transf.fiducial_radius_from_shape(10.0, 4)


@pytest.mark.parametrize(
    ("geographic_az", "expected_corsika_az"),
    [
        (0.0, 180.0),
        (90.0, 90.0),
        (180.0, 0.0),
        (270.0, 270.0),
        (360.0, 180.0),
        (450.0, 90.0),
        (-180.0, 0.0),
        (45.7, 134.3),
        (135.3, 44.7),
        (225.5, 314.5),
        (315.8, 224.2),
    ],
)
def test_geographic_to_corsika_azimuth(geographic_az, expected_corsika_az):
    """geographic_to_corsika_azimuth converts geographic azimuth to CORSIKA azimuth."""
    assert transf.geographic_to_corsika_azimuth(geographic_az) == pytest.approx(expected_corsika_az)


def test_geographic_to_corsika_azimuth_is_self_inverse():
    """Applying the conversion twice must return the original value (modulo 360)."""
    for az in [0.0, 45.0, 90.0, 135.0, 180.0, 270.0, 315.0]:
        assert transf.geographic_to_corsika_azimuth(
            transf.geographic_to_corsika_azimuth(az)
        ) % 360 == pytest.approx(az % 360)
