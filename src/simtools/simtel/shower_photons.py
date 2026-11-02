"""Input-plane frame for CORSIKA photon bunches traced for sim_telarray.

CORSIKA photon bunches are given in the array ground frame (x north, y west,
z up) at the horizontal plane through the telescope sphere centre. This module
describes that input plane as seen by the photon-bunch tracing: the rotation
from the input frame to the telescope frame, and the scale and plane height
of the input coordinates.
"""

import math
from dataclasses import dataclass

import numpy as np

from simtools.simtel.photon_bunch_file import BUNCH_DTYPE, camera_bunches
from simtools.utils import geometry

CORSIKA_LENGTH_UNIT_M = 0.01
# Height of the CORSIKA bunch plane (horizontal plane through the telescope sphere centre)
# in input units. Bunches are projected from this plane up to the input plane.
CORSIKA_PLANE_Z = 0.0
CM_PER_M = 100.0
# Margin above the highest incoming obscurer for the minimum input plane height [m].
INPUT_PLANE_MARGIN_M = 1.0
SPEED_OF_LIGHT_CM_PER_NS = 29.9792458
_TRACE_INPUT_KEYS = ("x", "y", "cx", "cy", "time_ns", "zem", "size", "wavelength_nm", "bunch_id")


@dataclass(frozen=True)
class ShowerBunchFrame:
    """Frame of the photon-bunch input plane.

    Parameters
    ----------
    rotation: numpy.ndarray
        Row-major 3x3 rotation matrix mapping input coordinates to the telescope
        frame, p_telescope = rotation @ p_input + offset_m.
    offset_m: numpy.ndarray
        Length-3 offset of the input origin in the telescope frame [m].
    length_unit_m: float
        Length of one input coordinate unit [m].
    plane_z: float
        Height of the input plane along the input z axis [input units]; the height
        above the telescope sphere centre, 100 * input_plane_height_m [cm].
    """

    rotation: np.ndarray
    offset_m: np.ndarray
    length_unit_m: float
    plane_z: float


def minimum_input_plane_height_m(optical_model, theta_deg, *, margin_m=INPUT_PLANE_MARGIN_M):
    """Return the minimum safe input plane height above the telescope sphere centre [m].

    The input plane must lie above every incoming obscurer of the optical model, or photons
    starting below an obscurer never meet it and its shadow is lost. The height is the top of the
    highest incoming obscurer (``optical_model.max_incoming_obscurer_z_m()``, telescope frame) plus
    ``margin_m``. The plane height along the telescope axis is the input height times
    cos(theta), so the minimum input height is that value divided by cos(theta). Without incoming
    obscurers the top is taken as the sphere centre.

    Parameters
    ----------
    optical_model: obdeect.OpticalModel
        Optical model of the telescope at the zenith angle of the bunches.
    theta_deg: float
        Telescope zenith angle [deg].
    margin_m: float
        Margin above the highest incoming obscurer [m].

    Returns
    -------
    float
        Minimum input plane height above the telescope sphere centre [m].
    """
    top_m = optical_model.max_incoming_obscurer_z_m()
    top_m = 0.0 if top_m is None else float(top_m)
    return (top_m + margin_m) / math.cos(math.radians(theta_deg))


def shower_bunch_frame(theta, phi, *, input_plane_height_m):
    """Return the frame of the photon-bunch input plane for CORSIKA bunches.

    The input frame is the array ground frame (x north, y west, z up). The
    columns of the rotation are the telescope-frame images of the three input
    unit vectors, so rotation[:, j] is the image of the j-th input axis.

    The axes offset of a telescope (``axes_offsets``) is applied by the optical model, which
    displaces the input origin in the telescope frame; the frame offset here is therefore zero.

    Parameters
    ----------
    theta: float
        Telescope zenith angle [deg].
    phi: float
        Telescope azimuth in the array frame [deg].
    input_plane_height_m: float
        Height of the input plane above the telescope sphere centre [m]. It must be
        above the incoming obscurers (see minimum_input_plane_height_m).

    Returns
    -------
    ShowerBunchFrame
        Rotation, offset, length unit and plane height (input units) of the input plane.
    """
    rotation = np.column_stack(
        [geometry.ground_direction_to_telescope_frame(unit, theta, phi) for unit in np.eye(3)]
    ).astype(np.float64)
    return ShowerBunchFrame(
        rotation=rotation,
        offset_m=np.zeros(3, dtype=np.float64),
        length_unit_m=CORSIKA_LENGTH_UNIT_M,
        plane_z=CM_PER_M * float(input_plane_height_m),
    )


def project_to_input_plane(bunches, plane_z):
    """Project photon bunches from the CORSIKA plane up to the input plane, along their direction.

    A bunch at (x, y, CORSIKA_PLANE_Z) with direction (cx, cy, cz), cz < 0 (downwards), is moved
    along its direction to height ``plane_z``. The arrival time is corrected for the path
    between the two planes: the photon reaches the input plane earlier by the path length
    divided by the speed of light.

    Parameters
    ----------
    bunches: numpy.ndarray or mapping
        Bunches with fields x, y, cx, cy and time_ns (input units: cm and ns).
    plane_z: float
        Height of the input plane [input units, cm].

    Returns
    -------
    dict
        Keys x, y and time_ns as float64 arrays; x, y at the input plane and time_ns corrected.
    """
    cx = np.asarray(bunches["cx"], dtype=np.float64)
    cy = np.asarray(bunches["cy"], dtype=np.float64)
    cz = -np.sqrt(np.maximum(0.0, 1.0 - cx**2 - cy**2))
    height = float(plane_z) - CORSIKA_PLANE_Z
    s = height / cz
    return {
        "x": np.asarray(bunches["x"], dtype=np.float64) + s * cx,
        "y": np.asarray(bunches["y"], dtype=np.float64) + s * cy,
        "time_ns": np.asarray(bunches["time_ns"], dtype=np.float64)
        - (height / np.abs(cz)) / SPEED_OF_LIGHT_CM_PER_NS,
    }


def camera_bunches_of_event(
    model, bunches, frame, *, seed, threads, dual_mirror, camera_rotation_deg
):
    """Trace the photon bunches of one event and return them as camera photon bunches."""
    if len(bunches["x"]) == 0:
        return np.empty(0, dtype=BUNCH_DTYPE)
    projected = project_to_input_plane(bunches, frame.plane_z)
    traced = model.trace_bunches(
        {
            **{key: bunches[key] for key in _TRACE_INPUT_KEYS if key not in projected},
            **projected,
        },
        seed=seed,
        frame_rotation=frame.rotation,
        frame_offset_m=frame.offset_m,
        length_unit_m=frame.length_unit_m,
        plane_z=frame.plane_z,
        wavelength_law=(bunches["wavelength_min_nm"], bunches["wavelength_max_nm"]),
        roulette=True,
        normalise_transmission=False,
        threads=threads,
    )
    if len(traced["bunch_index"]) == 0:
        return np.empty(0, dtype=BUNCH_DTYPE)
    emission_altitude_m = bunches["zem"][traced["bunch_index"]].astype(np.float64) / CM_PER_M
    photons = {
        "position_m": traced["position_m"],
        "direction": traced["direction"],
        "time_ns": traced["time_ns"],
        "weight": traced["weight"],
        "wavelength_nm": traced["wavelength_nm"],
    }
    return camera_bunches(
        photons,
        dual_mirror=dual_mirror,
        emission_altitude_m=emission_altitude_m,
        camera_rotation_deg=camera_rotation_deg,
    )
