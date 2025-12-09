"""A collection of functions related to geometrical transformations."""

import math

import astropy.units as u
import numpy as np
from astropy.units import UnitsError


@u.quantity_input(rotation_angle_phi=u.rad, rotation_angle_theta=u.rad)
def rotate(x, y, rotation_around_z_axis, rotation_around_y_axis=0):
    """
    Rotate the x and y coordinates of the telescopes.

    The two rotations are:

    - rotation_angle_around_z_axis gives the rotation on the observation plane (x, y)
    - rotation_angle_around_y_axis allows to rotate the observation plane in space.

    The function returns the rotated x and y values in the same unit given.
    The direction of rotation of the elements in the plane is counterclockwise, i.e.,
    the rotation of the coordinate system is clockwise.

    Parameters
    ----------
    x: numpy.array or list
        x positions of the entries (e.g. telescopes), usually in meters.
    y: numpy.array or list
        y positions of the entries (e.g. telescopes), usually in meters.
    rotation_angle_around_z_axis: astropy.units.rad
        Angle to rotate the array in the observation plane (around z axis) in radians.
    rotation_angle_around_y_axis: astropy.units.rad
        Angle to rotate the observation plane around the y axis in radians.

    Returns
    -------
    2-tuple of list
        x and y positions of the rotated entry (e.g. telescopes) positions.

    Raises
    ------
    TypeError:
        If type of x and y parameters are not valid.
    RuntimeError:
        If the length of x and y are different.
    UnitsError:
        If the unit of x and y are different.
    """
    allowed_types = (list, np.ndarray, u.Quantity, float, int)
    if not all(isinstance(variable, (allowed_types)) for variable in [x, y]):
        raise TypeError("x and y types are not valid! Cannot perform transformation.")

    if not isinstance(x, list | np.ndarray):
        x = [x]
    if not isinstance(y, list | np.ndarray):
        y = [y]

    if (
        np.sum(
            np.array([isinstance(x, type_now) for type_now in allowed_types[:-2]])
            * np.array([isinstance(y, type_now) for type_now in allowed_types[:-2]])
        )
        == 0
    ):
        raise TypeError("x and y are not from the same type! Cannot perform transformation.")

    if len(x) != len(y):
        raise RuntimeError(
            "Cannot perform coordinate transformation when x and y have different lengths."
        )
    if all(isinstance(variable, (u.Quantity)) for variable in [x, y]):
        if not isinstance(x[0].unit, type(y[0].unit)):
            raise UnitsError(
                "Cannot perform coordinate transformation when x and y have different units."
            )

    x_trans = np.cos(rotation_around_y_axis) * (
        x * np.cos(rotation_around_z_axis) - y * np.sin(rotation_around_z_axis)
    )
    y_trans = x * np.sin(rotation_around_z_axis) + y * np.cos(rotation_around_z_axis)

    return x_trans, y_trans


def calculate_circular_mean(angles):
    """
    Calculate circular mean of angles in radians.

    Parameters
    ----------
    angles: numpy.array
        Array of angles in radians.

    Returns
    -------
    float
        Circular mean of the angles.
    """
    sin_sum = np.sum(np.sin(angles))
    cos_sum = np.sum(np.cos(angles))
    return np.arctan2(sin_sum, cos_sum)


@u.quantity_input(angle_max=u.rad, angle_min=u.rad)
def solid_angle(angle_max, angle_min=0 * u.rad):
    """
    Calculate the solid angle subtended by a given range of angles.

    Parameters
    ----------
    angle_max: astropy.units.Quantity
        The maximum angle for which to calculate the solid angle.
    angle_min: astropy.units.Quantity
        The minimum angle for which to calculate the solid angle (default is 0 rad).

    Returns
    -------
    astropy.units.Quantity
        The solid angle subtended by the given range of angles (in steradians).
    """
    return 2 * np.pi * (np.cos(angle_min.to("rad")) - np.cos(angle_max.to("rad"))) * u.sr


def transform_ground_to_shower_coordinates(x_ground, y_ground, z_ground, azimuth, altitude):
    """
    Transform ground to shower coordinates.

    Assume ground to be of type 'North-West-Up' (NWU) coordinates.

    Parameters
    ----------
    x_ground: numpy.array
        Ground x coordinate.
    y_ground: numpy.array
        Ground y coordinate.
    z_ground: numpy.array
        Ground z coordinate.
    azimuth: numpy.array
        Azimuth angle of the shower (in radians).
    altitude: numpy.array
        Altitude angle of the shower (in radians).

    Returns
    -------
    tuple
        Transformed shower coordinates (x', y', z').
    """
    x, y, z, az, alt = np.broadcast_arrays(x_ground, y_ground, z_ground, azimuth, altitude)

    ca, sa = np.cos(az), np.sin(az)
    cz, sz = np.sin(alt), np.cos(alt)

    x_s = ca * cz * x - sa * y + ca * sz * z
    y_s = sa * cz * x + ca * y + sa * sz * z
    z_s = -sz * x + cz * z

    return x_s, y_s, z_s


def fiducial_radius_from_shape(width, shape):
    """
    Calculate minimum radius including different geometrical shapes.

    Assumes definition of shapes as in 'camera_body_shape' model parameter:

    - circle: shape = 0, width is diameter
    - hexagon: shape = 1 or 3, width is flat-to-flat distance
    - square: shape = 2, width is side length

    Parameters
    ----------
    width : float
        Characteristic width
    shape : int
        Geometrical shape parameter

    Returns
    -------
    float
        Minimum fiducial radius
    """
    if shape == 0:
        return width / 2.0
    if shape == 2:
        return width / math.sqrt(2.0)
    if shape in (1, 3):
        return width / math.sqrt(3.0)
    raise ValueError(
        f"Unknown shape value {shape}. Valid values are: 0 (circle), 1 or 3 (hexagon), 2 (square)."
    )
