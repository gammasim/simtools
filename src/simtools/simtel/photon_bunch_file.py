"""Photon bunch files for sim_telarray (CORSIKA IACT EventIO format).

sim_telarray reads Cherenkov photons as photon bunches in the EventIO format
of the CORSIKA IACT interface. This module writes such files for light that
simtools makes itself (e.g. calibration light traced by obdeect), with the
minimal set of objects sim_telarray needs, filled as the sim_telarray
LightEmission programs (ff-1m, xyzls) fill them for artificial light.

One telescope at the origin of the array frame; one photon bunch object per
event.
"""

import gzip
import struct
from pathlib import Path

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
# artificial light. Word 77 carries the atmospheric profile number times 1024:
# sim_telarray reads atmprof<number>.dat from its include path.
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


def _object(object_type, identifier, payload, top_level=True, only_subobjects=False):
    """Encode one EventIO object (version 0)."""
    header = struct.pack("<IiI", object_type, identifier, len(payload) | (only_subobjects << 30))
    return (struct.pack("<I", _SYNC) if top_level else b"") + header + payload


def _block(name, words):
    """Encode a CORSIKA header block: length, name and 272 floats (1-based word numbers)."""
    values = np.zeros(_BLOCK_WORDS - 1, dtype="<f4")
    for index, value in words.items():
        values[index - 2] = value
    return struct.pack("<i", _BLOCK_WORDS) + name + values.tobytes()


def write_photon_bunches(
    path, events, *, observation_level_m, telescope_radius_m, run_number=1, atmosphere=1
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
    run_number: int
        Run number.
    atmosphere: int
        Number of the atmospheric profile (file atmprof<number>.dat).

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
        for count, bunches in enumerate(events, start=1):
            bunches = np.ascontiguousarray(bunches, dtype=BUNCH_DTYPE)
            event = _EVENT_WORDS | {2: count, 5: level_cm, 7: -level_cm, 44: run_number}
            event |= {48: level_cm, 77: 1024 * atmosphere + _EVENT_FLAGS}
            output.write(_object(1202, count, _block(b"EVTH", event)))
            output.write(_object(1203, 0, struct.pack("<i3f", 1, 0.0, 0.0, 0.0)))
            photons = struct.pack("<hhfi", 0, 0, float(bunches["photons"].sum()), len(bunches))
            bunch_object = _object(1205, 0, photons + bunches.tobytes(), top_level=False)
            output.write(_object(1204, 0, bunch_object, only_subobjects=True))
            output.write(_object(1209, count, _block(b"EVTE", {2: count})))
        run_end = struct.pack("<i", 3) + b"RUNE" + struct.pack("<2f", run_number, count)
        output.write(_object(1210, run_number, run_end))
    return count
