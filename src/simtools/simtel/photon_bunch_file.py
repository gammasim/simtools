"""Photon bunch files for sim_telarray (CORSIKA IACT EventIO format).

sim_telarray reads Cherenkov photons as photon bunches in the EventIO format
of the CORSIKA IACT interface. This module writes such files for light that
simtools makes itself (e.g. calibration light traced by obdeect), with the
minimal set of objects sim_telarray needs, filled as the sim_telarray
LightEmission programs (ff-1m, xyzls) fill them for artificial light.

One telescope at the origin of the array frame; one photon bunch object per
event. ``camera_bunches`` hands photons at the camera entrance to sim_telarray
with its optics bypassed (``Bypass_Optics``).
"""

import gzip
import math
import struct
from pathlib import Path

import astropy.units as u
import numpy as np

BUNCH_DTYPE = np.dtype(
    [
        ("x", "<f4"),  # [cm], at the horizontal plane through the telescope sphere centre
        ("y", "<f4"),
        ("cx", "<f4"),  # Direction cosines; the photon travels downwards.
        ("cy", "<f4"),
        ("time", "<f4"),  # [ns]
        ("zem", "<f4"),  # Emission altitude a.s.l. [cm]
        ("photons", "<f4"),  # Photons in the bunch (below 1: probability)
        ("wavelength", "<f4"),  # [nm]
    ]
)

_SYNC = 0xD41F8A37
_BLOCK_WORDS = 273  # CORSIKA run and event header blocks.
_CORSIKA_VERSION = 6.999
_DATE = 261022.0
_ENERGY_GEV = 1e-9
# Event header words (1-based) as the LightEmission programs write them for
# artificial light. Word 77 carries the atmospheric profile number times 1024;
# the profile itself is written into the file.
_EVENT_WORDS = {
    3: 1.0,
    4: _ENERGY_GEV,
    10: -1.0,
    12: np.pi,
    46: _CORSIKA_VERSION,
    45: _DATE,
    47: 1.0,
    59: _ENERGY_GEV,
    60: _ENERGY_GEV,
    61: 1.0,
    62: 0.106,
    63: 0.000512,
    64: _ENERGY_GEV,
    71: 20.0,
    75: 2.0,
    76: 3.0,
    80: 3.0,
    83: 180.0,
    84: 180.0,
    85: 1.0,
    86: 1.0,
    87: 1.0,
    95: 1.0,
    96: 200.0,
    97: 1000.0,
    98: 1.0,
}
_EVENT_FLAGS = 187  # Low bits of word 77.
_PROFILE_ID = 99


def bypass_optics_level(dual_mirror):
    """Return the sim_telarray ``Bypass_Optics`` value for photons at the camera entrance."""
    return 2 if dual_mirror else 1


def camera_bunches(photons, *, dual_mirror, emission_altitude_m, camera_rotation_deg=0.0):
    """
    Photon bunches of photons at the camera entrance, for sim_telarray with its optics bypassed.

    The photons cross the camera plane (through the focal-surface vertex,
    normal to the optical axis) in the telescope frame (+z along the optical
    axis, towards the sky).

    - Single-mirror telescopes (``Bypass_Optics=1``): sim_telarray reads a
      bunch (x, y) as the camera position M (x, y) with
      M = [[cos r, -sin r], [-sin r, -cos r]] (camera rotation r), whereas its
      own optics give R(-r) (x, y) of the telescope frame. Since
      M R(-r) = diag(1, -1), the telescope frame is turned by pi about x, for
      any camera rotation: (x, y, z) -> (x, -y, -z).
    - Dual-mirror telescopes (``Bypass_Optics=2``): sim_telarray reads bunches
      in its camera frame, the telescope frame rotated by r about the optical
      axis; the light travels to -z.

    Parameters
    ----------
    photons: dict
        ``position_m`` (n, 2 or 3), ``direction`` (n, 3), ``time_ns``,
        ``weight`` and ``wavelength_nm`` of the photons.
    dual_mirror: bool
        Dual-mirror telescope.
    emission_altitude_m: float or numpy.ndarray
        Altitude a.s.l. sim_telarray takes as the emission point.
    camera_rotation_deg: float
        Camera rotation (dual-mirror telescopes) [deg].

    Returns
    -------
    numpy.ndarray
        Photon bunches (``BUNCH_DTYPE``); the photons per bunch are the weights.
    """
    position, direction = np.asarray(photons["position_m"]), np.asarray(photons["direction"])
    x, y, cx, cy = position[:, 0], position[:, 1], direction[:, 0], direction[:, 1]
    if dual_mirror:
        rotation = math.radians(camera_rotation_deg)
        cos_r, sin_r = math.cos(rotation), math.sin(rotation)
        x, y = x * cos_r + y * sin_r, y * cos_r - x * sin_r
        cx, cy = cx * cos_r + cy * sin_r, cy * cos_r - cx * sin_r
    else:
        y, cy = -y, -cy
    bunches = np.empty(len(position), dtype=BUNCH_DTYPE)
    bunches["x"], bunches["y"] = 100.0 * x, 100.0 * y
    bunches["cx"], bunches["cy"] = cx, cy
    bunches["time"] = photons["time_ns"]
    bunches["zem"] = 100.0 * np.asarray(emission_altitude_m)
    bunches["photons"] = photons["weight"]
    bunches["wavelength"] = photons["wavelength_nm"]
    return bunches


def _object_header(
    object_type, identifier, payload_length, top_level=True, only_subobjects=False, version=0
):
    """Encode the header of one EventIO object with a payload of ``payload_length`` bytes."""
    header = struct.pack(
        "<IiI",
        object_type | (version << 20),
        identifier,
        payload_length | (only_subobjects << 30),
    )
    return (struct.pack("<I", _SYNC) if top_level else b"") + header


def _object(object_type, identifier, payload, top_level=True, only_subobjects=False, version=0):
    """Encode one EventIO object."""
    return (
        _object_header(
            object_type,
            identifier,
            len(payload),
            top_level=top_level,
            only_subobjects=only_subobjects,
            version=version,
        )
        + payload
    )


def _count(value):
    """Encode an EventIO variable-length unsigned count."""
    for size, marker in ((1, 0x00), (2, 0x80), (3, 0xC0), (4, 0xE0)):
        if value < 1 << (8 * size - size):
            return (value | (marker << (8 * size - 8))).to_bytes(size, "big")
    return b"\xf0" + value.to_bytes(4, "big")


def _atmospheric_profile(table, observation_level_m, identifier):
    """Encode an atmospheric profile (EventIO type 1216, as CORSIKA writes it).

    The table has the columns of the ``atmospheric_profile`` model parameter:
    altitude, density, thickness and refractive_index (n - 1).
    """
    name = str(table.meta.get("parameter_name", "atmospheric_profile")).encode()
    columns = [
        u.Quantity(table["altitude"], u.km).to_value(u.km),
        u.Quantity(table["density"], u.g / u.cm**3).to_value(u.g / u.cm**3),
        u.Quantity(table["thickness"], u.g / u.cm**2).to_value(u.g / u.cm**2),
        np.asarray(table["refractive_index"], dtype=float),
    ]
    rows = np.column_stack(columns).astype("<f8")
    payload = (
        _count(len(name))
        + name
        + struct.pack("<d", 100.0 * observation_level_m)
        + _count(len(rows))
        + rows.tobytes()
        + _count(0)  # No 5-layer parameterisation.
    )
    return _object(1216, identifier, payload, version=1)


def _block(name, words):
    """Encode a CORSIKA header block: length, name and 272 floats (1-based word numbers)."""
    values = np.zeros(_BLOCK_WORDS - 1, dtype="<f4")
    for index, value in words.items():
        values[index - 2] = value
    return struct.pack("<i", _BLOCK_WORDS) + name + values.tobytes()


def write_photon_bunches(
    path,
    events,
    *,
    observation_level_m,
    telescope_radius_m,
    atmospheric_profile,
    run_number=1,
):
    """Write photon bunches of one telescope, one array of bunches per event.

    Parameters
    ----------
    path: str or Path
        Output file; gzip compressed if its name ends with ``.gz``.
    events: iterable of numpy.ndarray
        Photon bunches of each event, structured as ``BUNCH_DTYPE``.
    observation_level_m: float
        Altitude of the observation level (CORSIKA) a.s.l. [m].
    telescope_radius_m: float
        Radius of the telescope's fiducial sphere [m].
    atmospheric_profile: astropy.table.Table
        Atmospheric profile (``atmospheric_profile`` model parameter), written
        into the file.
    run_number: int
        Run number.

    Returns
    -------
    int
        Number of events written.
    """
    level_cm = 100.0 * observation_level_m
    run = {2: run_number, 3: _DATE, 4: _CORSIKA_VERSION, 5: 1, 6: level_cm}
    run |= {17: _ENERGY_GEV, 18: _ENERGY_GEV}
    path = Path(path)
    count = 0
    # Level 1: as small as the default level for photon bunches, 25 times faster.
    with (
        gzip.open(path, "wb", compresslevel=1)
        if path.suffix == ".gz"
        else open(path, "wb") as output
    ):
        output.write(_object(1200, run_number, _block(b"RUNH", run)))
        output.write(
            _object(1201, 0, struct.pack("<i4f", 1, 0.0, 0.0, 0.0, 100.0 * telescope_radius_m))
        )
        output.write(_atmospheric_profile(atmospheric_profile, observation_level_m, _PROFILE_ID))
        for count, bunches in enumerate(events, start=1):
            bunches = np.ascontiguousarray(bunches, dtype=BUNCH_DTYPE)
            event = _EVENT_WORDS | {2: count, 5: level_cm, 7: -level_cm, 44: run_number}
            event |= {48: level_cm, 77: 1024 * _PROFILE_ID + _EVENT_FLAGS}
            output.write(_object(1202, count, _block(b"EVTH", event)))
            output.write(_object(1203, 0, struct.pack("<i3f", 1, 0.0, 0.0, 0.0)))
            photons = struct.pack("<hhfi", 0, 0, float(bunches["photons"].sum()), len(bunches))
            payload_length = len(photons) + bunches.nbytes
            bunch_header = _object_header(1205, 0, payload_length, top_level=False)
            output.write(
                _object_header(1204, 0, len(bunch_header) + payload_length, only_subobjects=True)
            )
            output.write(bunch_header)
            output.write(photons)
            output.write(memoryview(bunches))
            output.write(_object(1209, count, _block(b"EVTE", {2: count})))
        run_end = struct.pack("<i", 3) + b"RUNE" + struct.pack("<2f", run_number, count)
        output.write(_object(1210, run_number, run_end))
    return count
