"""A collection of functions related to geometrical transformations."""

import math

import astropy.units as u
import numpy as np
from astropy.units import UnitsError


@u.quantity_input(rotation_around_z_axis=u.rad, rotation_around_y_axis=u.rad)
def rotate(x, y, rotation_around_z_axis, rotation_around_y_axis=0 * u.rad):
    """
    Rotate the x and y coordinates of the telescopes.

    The two rotations are:

    - rotation_around_z_axis gives the rotation on the observation plane (x, y)
    - rotation_around_y_axis allows to rotate the observation plane in space.

    The function returns the rotated x and y values in the same unit given.
    The direction of rotation of the elements in the plane is counterclockwise, i.e.,
    the rotation of the coordinate system is clockwise.

    Parameters
    ----------
    x: numpy.array or list
        x positions of the entries (e.g. telescopes), usually in meters.
    y: numpy.array or list
        y positions of the entries (e.g. telescopes), usually in meters.
    rotation_around_z_axis: astropy.units.rad
        Angle to rotate the array in the observation plane (around z axis) in radians.
    rotation_around_y_axis: astropy.units.rad
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


def project_ground_to_corsika_shower_coordinates(
    x_ground, y_ground, z_ground, azimuth_from_north_east, altitude
):
    """
    Transform ground to shower coordinates following the CORSIKA/sim_telarray convention.

    This function applies the same horizontal coordinate transformation used
    internally by sim_telarray for shower-oriented coordinates.

    Input ground coordinates follow the CORSIKA/sim_telarray ground system:

    - x_ground : points North
    - y_ground : points West
    - z_ground : points Up

    The input azimuth is assumed to follow the standard astronomical convention:

    - 0 rad   = North
    - pi/2    = East

    Internally, the azimuth is converted to the CORSIKA 'phi' convention used
    in 'iact.c':

    - phi = 0       : shower propagates toward North
    - phi = pi/2    : shower propagates toward West

    The transformation corresponds to the inverse of the projection applied
    in sim_telarray when converting shower coordinates into ground coordinates.

    Parameters
    ----------
    x_ground: numpy.array
        Ground x coordinate.
    y_ground: numpy.array
        Ground y coordinate.
    z_ground: numpy.array
        Ground z coordinate.
    azimuth_from_north_east: numpy.array
        Azimuth angle of the shower (in radians).
    altitude: numpy.array
        Altitude angle of the shower (in radians).

    Returns
    -------
    tuple
        Transformed shower coordinates (x', y', z').
    """
    x, y, z, az, alt = np.broadcast_arrays(
        x_ground, y_ground, z_ground, azimuth_from_north_east, altitude
    )

    # Convert to CORSIKA phi convention used in iact.c:
    # 0 = moves North, 90 deg = moves West
    phi = np.mod(np.pi - az, 2.0 * np.pi)

    ca, sa = np.cos(phi), np.sin(phi)
    cos_zenith = np.sin(alt)  # pi/2 - altitude

    x_s = (ca * ca * cos_zenith + sa * sa) * x + ca * sa * (cos_zenith - 1.0) * y
    y_s = ca * sa * (cos_zenith - 1.0) * x + (sa * sa * cos_zenith + ca * ca) * y
    z_s = z

    return x_s, y_s, z_s


def geographic_to_corsika_azimuth(az_deg):
    """Convert geographic arrival azimuth to CORSIKA propagation azimuth.

    The input is an astronomical/geographic azimuth: the direction from which
    the primary arrives, measured clockwise from geographic North toward East.

    CORSIKA phi describes the direction toward which the primary propagates
    and increases from North toward West. The conversion therefore consists of:

    1. Negating the angle because the two azimuth conventions increase in
       opposite directions.
    2. Adding 180 degrees to convert an arrival direction into the opposite
       propagation direction.

    No geomagnetic-field alignment correction is applied here.

    Parameters
    ----------
    az_deg : float
        Geographic arrival azimuth in degrees, measured clockwise from
        geographic North toward East.

    Returns
    -------
    float
        CORSIKA propagation azimuth in degrees, before geomagnetic-field
        alignment correction.
    """
    return (-az_deg + 180.0) % 360.0


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


def telescope_fiducial_sphere_radius(
    radial_extent,
    axial_extent,
    axes_offsets,
    mirror_offset=0 * u.m,
    zenith_range=(0, 90) * u.deg,
    margin=0 * u.m,
):
    """Enclose optical elements for a continuous range of telescope pointings.

    Each element is bounded by a cylinder about the optical axis. Use bounds
    containing the entire surface, including panel corners and surface sag,
    for every relevant primary, secondary and camera element. The result is
    conservative for these enclosing cylinders, rather than a minimum sphere
    for the individual optical surfaces. Azimuth rotations preserve distance
    from the CORSIKA sphere center.

    Parameters
    ----------
    radial_extent : astropy.units.Quantity
        One maximum radial distance from the optical axis per element.
    axial_extent : astropy.units.Quantity
        Array of shape (number of elements, 2) giving each element's minimum
        and maximum axial coordinate relative to the mirror backplane.
    axes_offsets : astropy.units.Quantity
        The two sim_telarray offsets: azimuth-to-altitude axis, followed by
        altitude-to-optical axis. Positive offsets follow ``axes_offsets``.
    mirror_offset : astropy.units.Quantity, optional
        sim_telarray mirror-backplane offset from the altitude axis.
    zenith_range : astropy.units.Quantity, optional
        Minimum and maximum zenith angle, within [0, 180] degrees.
    margin : astropy.units.Quantity, optional
        Nonnegative extra radial clearance. No clearance is added by default.

    Returns
    -------
    astropy.units.Quantity
        Required radius about the CORSIKA telescope position, in meters.

    Raises
    ------
    ValueError
        If bounds are empty, malformed, non-finite or outside their allowed ranges.

    Notes
    -----
    The offsets follow ``tel_newdir`` and ``transform_off`` in sim_telarray.
    The calculation checks the endpoints and analytic stationary points of
    the distance over zenith angle; it does not sample a pointing grid.
    """
    radii = np.atleast_1d(radial_extent.to_value(u.m))
    heights = np.asarray(axial_extent.to_value(u.m))
    offsets = np.asarray(axes_offsets.to_value(u.m))
    angles = np.asarray(zenith_range.to_value(u.rad))
    backplane_offset = float(mirror_offset.to_value(u.m))
    clearance = float(margin.to_value(u.m))
    _validate_fiducial_sphere_bounds(radii, heights, offsets, angles)
    if not np.isfinite(backplane_offset) or not np.isfinite(clearance) or clearance < 0:
        raise ValueError("Mirror offset and nonnegative margin must be finite.")

    # For each cylinder, extrema lie on its radial boundary and axial endpoints.
    x = np.repeat(np.stack((-radii, radii), axis=1), 2, axis=1).ravel() - offsets[1]
    z = np.tile(heights - backplane_offset, (1, 2)).ravel()
    azimuth_axis_offset = offsets[0]
    stationary = np.arctan2(azimuth_axis_offset * z, azimuth_axis_offset * x) % (2 * np.pi)
    candidates = np.array(
        [np.full_like(x, angles[0]), np.full_like(x, angles[1]), np.clip(stationary, *angles)]
    )
    ground_x = azimuth_axis_offset + x * np.cos(candidates) + z * np.sin(candidates)
    ground_z = -x * np.sin(candidates) + z * np.cos(candidates)
    return (float(np.max(np.hypot(ground_x, ground_z))) + clearance) * u.m


def _validate_fiducial_sphere_bounds(radii, heights, offsets, angles):
    """Validate cylindrical optical bounds and axis/pointing dimensions."""
    if radii.ndim != 1 or radii.size == 0 or heights.shape != (radii.size, 2):
        raise ValueError("Provide one radial extent and two axial bounds per optical element.")
    if offsets.shape != (2,) or angles.shape != (2,):
        raise ValueError("Provide two axes offsets and two zenith-angle bounds.")
    if not all(np.all(np.isfinite(values)) for values in (radii, heights, offsets, angles)):
        raise ValueError("Optical bounds, axes offsets and zenith angles must be finite.")
    if np.any(radii < 0) or np.any(heights[:, 0] > heights[:, 1]):
        raise ValueError("Radial extents must be nonnegative and axial bounds ordered.")
    if not 0 <= angles[0] <= angles[1] <= np.pi:
        raise ValueError("Zenith-angle bounds must be ordered within [0, 180] degrees.")
