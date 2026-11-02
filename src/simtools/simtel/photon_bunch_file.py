"""Photon bunch files for sim_telarray (CORSIKA IACT EventIO format).

sim_telarray reads Cherenkov photons as photon bunches in the EventIO format
of the CORSIKA IACT interface. This module writes such files for light that
simtools makes itself (e.g. calibration light traced by obdeect), with the
minimal set of objects sim_telarray needs, filled as the sim_telarray
LightEmission programs (ff-1m, xyzls) fill them for artificial light.

Telescopes are at the origin of the array frame and are identified by their telescope
id in the photon bunch objects. Each event has one event header and one photon bunch
object (EventIO type 1204) per telescope block. ``camera_bunches`` hands photons at the
camera entrance to sim_telarray with its optics bypassed (``Bypass_Optics``).
"""

import gzip
import io
import math
import os
import struct
import sys
from contextlib import contextmanager
from dataclasses import dataclass
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
_TELESCOPE_ID_MAX = 32767  # Telescope id is a 16-bit integer in the photon bunch payload.
_DEFAULT_TELESCOPE_ID = 0


@dataclass(frozen=True)
class PhotonBunchEvent:
    """
    Photon bunches of one event with a block of bunches per telescope.

    Parameters
    ----------
    number: int
        Event number, written to the event header and the event end.
    blocks: sequence of (int, numpy.ndarray)
        Telescope id and photon bunches (``BUNCH_DTYPE``) of each telescope block.
    """

    number: int
    blocks: tuple


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


@contextmanager
def _open_output(path):
    """
    Yield a binary output stream for a file path, ``"-"`` (stdout) or a binary stream.

    Only files opened here are closed; gzip compression applies to files ending in ``.gz``.
    """
    if not isinstance(path, str | os.PathLike):
        if isinstance(path, io.TextIOBase):
            raise TypeError(f"Photon bunches need a binary stream, not a text stream: {path!r}")
        if not callable(getattr(path, "write", None)):
            raise TypeError(f"Expected a file path or a binary stream, got {type(path).__name__}")
        yield path
    elif str(path) == "-":
        yield sys.stdout.buffer
    else:
        path = Path(path)
        if path.suffix == ".gz":
            # Level 1: as small as the default level for photon bunches, 25 times faster.
            with gzip.open(path, "wb", compresslevel=1) as output:
                yield output
        else:
            with open(path, "wb") as output:
                yield output


def _telescope_block(telescope_id, bunches):
    """Return the telescope id (int) and the photon bunches (``BUNCH_DTYPE``) of one block."""
    if isinstance(telescope_id, bool) or not isinstance(telescope_id, int | np.integer):
        raise TypeError(f"Telescope id must be an integer, got {telescope_id!r}")
    if not 0 <= telescope_id <= _TELESCOPE_ID_MAX:
        raise ValueError(f"Telescope id must be from 0 to {_TELESCOPE_ID_MAX}: {telescope_id}")
    return int(telescope_id), np.ascontiguousarray(bunches, dtype=BUNCH_DTYPE)


def _event_blocks(item, default_number):
    """
    Return the event number and the telescope blocks of one event of ``write_photon_bunches``.

    An event is a photon bunch array (telescope id 0, event number ``default_number``), a
    ``PhotonBunchEvent``, or a sequence of ``(telescope_id, bunches)`` pairs (one per telescope).
    """
    if isinstance(item, PhotonBunchEvent):
        return item.number, [_telescope_block(*block) for block in item.blocks]
    if isinstance(item, np.ndarray):
        return default_number, [_telescope_block(_DEFAULT_TELESCOPE_ID, item)]
    return default_number, [_telescope_block(*block) for block in item]


def _write_telescope_block(output, telescope_id, bunches, reuse=0):
    """Write one photon bunch object (1204) holding the bunches (1205) of one telescope.

    As in CORSIKA output, the identifier of the object is the core-reuse index of the block, and
    the identifier of the photon bunch object (1205) is ``1000 * array_id + telescope_id`` with
    the core-reuse index as array id.
    """
    if not 0 <= telescope_id < 1000:
        raise ValueError(f"Telescope id must be from 0 to 999 for photon bunches: {telescope_id}")
    photons = struct.pack("<hhfi", 0, telescope_id, float(bunches["photons"].sum()), len(bunches))
    payload_length = len(photons) + bunches.nbytes
    bunch_header = _object_header(
        1205, 1000 * reuse + telescope_id, payload_length, top_level=False
    )
    output.write(
        _object_header(1204, reuse, len(bunch_header) + payload_length, only_subobjects=True)
    )
    output.write(bunch_header)
    output.write(photons)
    output.write(memoryview(bunches))


def _corsika_card_payload(lines):
    """Return the payload of the CORSIKA input-card object (1212) for the given card lines.

    The payload holds the number of records, then each record as a 16-bit length and its text
    (the layout of CORSIKA's own card object); it is padded to a multiple of four bytes.
    sim_telarray reads the number of showers (``NSHOW``) from this card.
    """
    records = [line.encode("latin-1", errors="replace")[:65535] for line in lines]
    payload = struct.pack("<I", len(records)) + b"".join(
        struct.pack("<H", len(record)) + record for record in records
    )
    return payload + b"\0" * (-len(payload) % 4)


def _telescope_offsets_payload(reuses):
    """Return the payload of the telescope offsets object (1203) for ``reuses`` core reuses.

    One array per core reuse, all at zero offset (the photon bunches are relative to the
    telescope already): the number of arrays, a time offset, and the x and y offsets per array.
    """
    return struct.pack("<if", reuses, 0.0) + bytes(8 * reuses)


def _write_event_header(output, number, *, level_cm, run_number, reuses=1):
    """Write the event header (1202) and the telescope offsets (1203) of one shower."""
    event = _EVENT_WORDS | {2: number, 5: level_cm, 7: -level_cm, 44: run_number}
    event |= {48: level_cm, 77: 1024 * _PROFILE_ID + _EVENT_FLAGS}
    output.write(_object(1202, number, _block(b"EVTH", event)))
    output.write(_object(1203, 0, _telescope_offsets_payload(reuses)))


def _write_event_end(output, number):
    """Write the event end (1209) of one shower."""
    output.write(_object(1209, number, _block(b"EVTE", {2: number})))


def _write_telescope_blocks(output, blocks, reuse_counts):
    """Write one photon bunch object (1204) per telescope block.

    ``reuse_counts`` counts the blocks written so far per telescope id in the current shower;
    the reuse index of a block is its count (0 for the first block of a telescope).
    """
    for telescope_id, bunches in blocks:
        reuse = reuse_counts.get(telescope_id, 0)
        reuse_counts[telescope_id] = reuse + 1
        _write_telescope_block(output, telescope_id, bunches, reuse=reuse)


def write_photon_bunches(
    path,
    events,
    *,
    observation_level_m,
    telescope_radius_m,
    atmospheric_profile,
    run_number=1,
    corsika_card=None,
    reuses=1,
):
    """Write photon bunches of telescopes, one event per item.

    As CORSIKA does, each event has one event header and one photon bunch object (1204)
    per telescope block, holding the photon bunches (1205) with the telescope id of the block.

    Parameters
    ----------
    path: str, Path or binary stream
        Output file; gzip compressed if its name ends with ``.gz``. ``"-"`` writes to
        stdout, and a binary stream (e.g. ``io.BytesIO``) is written uncompressed and left open.
    events: iterable
        One item per event, each either

        - a numpy.ndarray of photon bunches (``BUNCH_DTYPE``) of telescope 0; the event
          number is the position of the event (1, 2, ...),
        - a sequence of ``(telescope_id, bunches)`` pairs, one per telescope block; the
          event number is the position of the event, or
        - a ``PhotonBunchEvent`` with its own event number.
    observation_level_m: float
        Altitude of the observation level (CORSIKA) a.s.l. [m].
    telescope_radius_m: float
        Radius of the telescope's fiducial sphere [m].
    atmospheric_profile: astropy.table.Table
        Atmospheric profile (``atmospheric_profile`` model parameter), written
        into the file.
    run_number: int
        Run number.
    corsika_card: list of str or None
        Lines of the CORSIKA input card (written as the card object, which holds the number of
        showers of the run for sim_telarray); None writes no card.
    reuses: int
        Number of core reuses per shower (one telescope offset array per reuse).

    Returns
    -------
    int
        Number of events written.
    """
    level_cm = 100.0 * observation_level_m
    run = {2: run_number, 3: _DATE, 4: _CORSIKA_VERSION, 5: 1, 6: level_cm}
    run |= {17: _ENERGY_GEV, 18: _ENERGY_GEV}
    count = 0
    with _open_output(path) as output:
        output.write(_object(1200, run_number, _block(b"RUNH", run)))
        if corsika_card is not None:
            output.write(_object(1212, 0, _corsika_card_payload(corsika_card)))
        output.write(
            _object(1201, 0, struct.pack("<i4f", 1, 0.0, 0.0, 0.0, 100.0 * telescope_radius_m))
        )
        output.write(_atmospheric_profile(atmospheric_profile, observation_level_m, _PROFILE_ID))
        # As in CORSIKA output, consecutive events with the same number are the core reuses of
        # one shower: they share one event header and one event end (sim_telarray counts showers
        # by event headers). The run end holds the number of showers.
        open_number = None
        reuse_counts = {}
        showers = 0
        for count, item in enumerate(events, start=1):
            number, blocks = _event_blocks(item, count)
            if number != open_number:
                if open_number is not None:
                    _write_event_end(output, open_number)
                _write_event_header(
                    output, number, level_cm=level_cm, run_number=run_number, reuses=reuses
                )
                open_number = number
                reuse_counts = {}
                showers += 1
            _write_telescope_blocks(output, blocks, reuse_counts)
            # Flushed per event: a consumer in sequential mode (see shower_route) is signalled
            # after each input object and must find its events in the pipe.
            output.flush()
        if open_number is not None:
            _write_event_end(output, open_number)
        run_end = struct.pack("<i", 3) + b"RUNE" + struct.pack("<2f", run_number, showers)
        output.write(_object(1210, run_number, run_end))
        output.flush()
    return count
