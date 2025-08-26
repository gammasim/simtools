"""A collection of functions related to geometrical transformations."""

import logging

import astropy.units as u
import numpy as np
from astropy.units import UnitsError

__all__ = [
    "convert_2d_to_radial_distr",
    "rotate",
]

_logger = logging.getLogger(__name__)


def convert_2d_to_radial_distr(hist_2d, xaxis, yaxis, bins=50, max_dist=1000):
    """
    Convert a 2d histogram of positions, e.g. photon positions on the ground, to a 1D distribution.

    Parameters
    ----------
    hist_2d: numpy.ndarray
        The histogram counts.
    xaxis: numpy.array
        The values of the x axis (histogram bin edges) on the ground.
    yaxis: numpy.array
        The values of the y axis (histogram bin edges) on the ground.
    bins: float
        Number of bins in distance.
    max_dist: float
       Maximum distance to consider in the 1D histogram, usually in meters.

    Returns
    -------
    np.array
        The values of the 1D histogram with size = int(max_dist/bin_size).
    np.array
        The bin edges of the 1D histogram with size = int(max_dist/bin_size) + 1.

    """
    # Check if the histogram will make sense
    bins_step = 2 * max_dist / bins  # in the 2d array, the positive and negative direction count.
    for axis in [xaxis, yaxis]:
        if (bins_step < np.diff(axis)).any():
            msg = (
                f"The histogram with number of bins {bins} and maximum distance of {max_dist} "
                f"resulted in a bin size smaller than the original array. Please adjust those "
                f"parameters to increase the bin size and avoid nan in the histogram values."
            )
            _logger.warning(msg)
            break

    grid_2d_x, grid_2d_y = np.meshgrid(xaxis[:-1], yaxis[:-1])  # [:-1], since xaxis and yaxis are
    # the hist bin_edges (n + 1).
    # radial_distance_map maps the distance to the center from each element in a square matrix.
    radial_distance_map = np.sqrt(grid_2d_x**2 + grid_2d_y**2)
    # The sorting and unravel_index give us the two indices for the position of the sorted element
    # in the original 2d matrix
    sorted_indices = np.unravel_index(
        np.argsort(radial_distance_map, axis=None), np.shape(radial_distance_map)
    )
    x_indices_sorted, y_indices_sorted = sorted_indices[0], sorted_indices[1]

    # We construct a 1D array with the histogram counts sorted according to the distance to the
    # center.
    hist_sorted = np.array(
        [hist_2d[i_x, i_y] for i_x, i_y in zip(x_indices_sorted, y_indices_sorted)]
    )
    distance_sorted = np.sort(radial_distance_map, axis=None)

    # For larger distances, we have more elements in a slice 'dr' in radius, hence, we need to
    # account for it using weights below.

    weights, radial_bin_edges = np.histogram(distance_sorted, bins=bins, range=(0, max_dist))
    histogram_1d = np.empty_like(weights, dtype=float)

    for i_radial, _ in enumerate(radial_bin_edges[:-1]):
        # Here we sum all the events within a radial interval 'dr' and then divide by the number of
        # bins that fit this interval.
        indices_to_sum = (distance_sorted >= radial_bin_edges[i_radial]) * (
            distance_sorted < radial_bin_edges[i_radial + 1]
        )
        if weights[i_radial] != 0:
            histogram_1d[i_radial] = np.sum(hist_sorted[indices_to_sum]) / weights[i_radial]
        else:
            histogram_1d[i_radial] = 0
    return histogram_1d, radial_bin_edges


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
