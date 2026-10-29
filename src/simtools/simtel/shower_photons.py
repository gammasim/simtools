"""Input-plane frame for CORSIKA photon bunches traced for sim_telarray.

CORSIKA photon bunches are given in the array ground frame (x north, y west,
z up) at the horizontal plane through the telescope sphere centre. This module
describes that input plane as seen by the photon-bunch tracing: the rotation
from the input frame to the telescope frame, and the scale and plane height
of the input coordinates.
"""

import logging
from dataclasses import dataclass

import numpy as np

from simtools.corsika.photon_bunches import read_photon_bunches
from simtools.simtel.photon_bunch_file import BUNCH_DTYPE, camera_bunches, write_photon_bunches
from simtools.utils import geometry

_logger = logging.getLogger(__name__)

CORSIKA_LENGTH_UNIT_M = 0.01
# Height of the CORSIKA bunch plane (horizontal plane through the telescope sphere centre)
# in input units. Bunches are projected from this plane up to the input plane.
CORSIKA_PLANE_Z = 0.0
CM_PER_M = 100.0
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


def shower_bunch_frame(theta, phi, *, input_plane_height_m):
    """Return the frame of the photon-bunch input plane for CORSIKA bunches.

    The input frame is the array ground frame (x north, y west, z up). The
    columns of the rotation are the telescope-frame images of the three input
    unit vectors, so rotation[:, j] is the image of the j-th input axis.

    The offset is zero because CORSIKA bunches are written per telescope, with
    x and y relative to the telescope sphere centre.

    Parameters
    ----------
    theta: float
        Telescope zenith angle [deg].
    phi: float
        Telescope azimuth in the array frame [deg].
    input_plane_height_m: float
        Height of the input plane above the telescope sphere centre [m]. It must be
        above the highest telescope element (20 m for the MST).

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


def write_camera_bunch_file(
    tracer,
    iact_file,
    telescope_model,
    output_path,
    *,
    theta_deg,
    phi_deg,
    dual_mirror,
    camera_rotation_deg,
    observation_level_m,
    telescope_radius_m,
    atmospheric_profile,
    seed,
    telescope_id,
    input_plane_height_m,
    threads=0,
):
    """
    Trace CORSIKA photon bunches to the camera entrance and write them for sim_telarray.

    The bunches, given on the CORSIKA plane through the telescope sphere centre, are projected
    along their direction up to the input plane at ``input_plane_height_m`` before tracing.

    Each event (telescope) of the IACT file is traced with the optical model of
    the tracer at the telescope zenith angle. The photons are written as photon
    bunches with the optics of the telescope bypassed, so that sim_telarray
    applies the atmosphere and the spectral quantum efficiency itself. Only the
    telescope transmission is applied here in absolute form: sim_telarray does not
    apply the on-axis value when the optics are bypassed.

    Parameters
    ----------
    tracer: ObdeectRayTracer
        Ray tracer providing ``optical_model(telescope_zenith, instrument_seed)``.
    iact_file: str or Path
        CORSIKA IACT EventIO file with photon bunches in long format.
    telescope_model: TelescopeModel
        Telescope model (used for logging).
    output_path: str or Path
        Output photon-bunch file; gzip compressed if its name ends with ``.gz``.
    theta_deg: float
        Telescope zenith angle [deg].
    phi_deg: float
        Telescope azimuth in the array frame [deg].
    dual_mirror: bool
        Dual-mirror telescope.
    camera_rotation_deg: float
        Camera rotation (dual-mirror telescopes) [deg].
    observation_level_m: float
        Altitude of the observation level a.s.l. [m].
    telescope_radius_m: float
        Radius of the telescope's fiducial sphere [m].
    atmospheric_profile: astropy.table.Table
        Atmospheric profile, written into the file.
    seed: int
        Seed of the instrument instance and of the random roulette and wavelength sampling.
    telescope_id: int
        Telescope of the IACT file whose photon bunches are written (one telescope per file).
    input_plane_height_m: float
        Height of the input plane above the telescope sphere centre [m]. It must be above the
        highest telescope element (20 m for the MST).
    threads: int
        Number of threads used for tracing (0: default of the optical model).

    Returns
    -------
    int
        Number of events written.
    """
    model = tracer.optical_model(theta_deg, seed)
    frame = shower_bunch_frame(theta_deg, phi_deg, input_plane_height_m=input_plane_height_m)
    events = (
        _camera_bunches_of_event(
            model,
            bunches,
            frame,
            seed=seed,
            threads=threads,
            dual_mirror=dual_mirror,
            camera_rotation_deg=camera_rotation_deg,
        )
        for bunches in read_photon_bunches(iact_file)
        if bunches["telescope_id"] == telescope_id
    )
    count = write_photon_bunches(
        output_path,
        events,
        observation_level_m=observation_level_m,
        telescope_radius_m=telescope_radius_m,
        atmospheric_profile=atmospheric_profile,
    )
    _logger.debug(
        "Wrote %d events of camera photon bunches for %s (theta=%.2f deg) to %s",
        count,
        telescope_model.name,
        theta_deg,
        output_path,
    )
    return count


def _camera_bunches_of_event(
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
