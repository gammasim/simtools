#!/usr/bin/python3

import gzip
import hashlib
import io
import math
import sys
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table
from eventio import EventIOFile

from simtools.simtel.photon_bunch_file import (
    BUNCH_DTYPE,
    PhotonBunchEvent,
    bypass_optics_level,
    camera_bunches,
    write_photon_bunches,
)

PHOTON_FILE_NAMES = ("photons.iact", "photons.iact.gz")
PROFILE_ALTITUDE_COLUMN = "altitude"
PROFILE_DENSITY_COLUMN = "density"
PROFILE_THICKNESS_COLUMN = "thickness"
PROFILE_REFRACTIVE_INDEX_COLUMN = "refractive_index"
ATMOSPHERE_REFRACTIVE_INDEX_KEY = "refractive_index_minus_1"


def _bunches(n, offset=0.0):
    bunches = np.zeros(n, dtype=BUNCH_DTYPE)
    bunches["x"] = np.arange(n) + offset
    bunches["y"] = -np.arange(n)
    bunches["cx"], bunches["cy"] = 0.01, -0.02
    bunches["time"] = 55.0
    bunches["zem"] = 216350.0
    bunches["photons"] = 0.5
    bunches["wavelength"] = 385.0
    return bunches


@pytest.fixture
def atmospheric_profile():
    return Table(
        {
            PROFILE_ALTITUDE_COLUMN: [0.0, 2.0, 10.0] * u.km,
            PROFILE_DENSITY_COLUMN: [1.2e-3, 1.0e-3, 4.0e-4] * u.g / u.cm**3,
            PROFILE_THICKNESS_COLUMN: [1030.0, 800.0, 270.0] * u.g / u.cm**2,
            PROFILE_REFRACTIVE_INDEX_COLUMN: [2.8e-4, 2.3e-4, 1.0e-4],
        }
    )


@pytest.mark.parametrize("name", PHOTON_FILE_NAMES)
def test_write_photon_bunches_reads_back(tmp_test_directory, atmospheric_profile, name):
    path = tmp_test_directory / name
    events = [_bunches(3), _bunches(2, offset=10.0)]

    count = write_photon_bunches(
        path,
        events,
        observation_level_m=2147.0,
        telescope_radius_m=2.0,
        atmospheric_profile=atmospheric_profile,
        run_number=7,
    )

    assert count == 2
    objects, bunches = [], []
    for item in EventIOFile(str(path)):
        objects.append(item.header.type)
        if item.header.type in (1200, 1202):
            words = item.read()
            header = (words[4:8], np.frombuffer(words[4:], dtype="<f4"))
            if item.header.type == 1200:
                run_header = header
            else:
                event_header = header
        elif item.header.type == 1201:
            telescopes = item.parse()
        elif item.header.type == 1216:
            atmosphere = item.parse()
        elif item.header.type == 1204:
            photons = next(iter(item))
            bunches.append(photons.parse()[0])
            assert photons.n_bunches == len(bunches[-1])
            assert photons.n_photons == pytest.approx(bunches[-1]["photons"].sum())
    assert objects == [1200, 1201, 1216, 1202, 1203, 1204, 1209, 1202, 1203, 1204, 1209, 1210]
    for written, read in zip(events, bunches):
        assert np.array_equal(written, read)
    # CORSIKA words (1-based): run number 2, observation level 6 (run) or 48 (event).
    assert run_header[0] == b"RUNH"
    assert run_header[1][1] == 7
    assert math.isclose(run_header[1][5], 214700.0, rel_tol=1e-6, abs_tol=1e-12)
    assert event_header[0] == b"EVTH"
    assert event_header[1][1] == 2
    assert math.isclose(event_header[1][47], 214700.0, rel_tol=1e-6, abs_tol=1e-12)
    # Word 77: the number of the profile in the file times 1024 plus option bits.
    assert int(event_header[1][76]) >> 10 == atmosphere["id"]
    assert math.isclose(atmosphere["obslevel"], 214700.0, rel_tol=1e-6, abs_tol=1e-12)
    assert atmosphere["name"] == b"atmospheric_profile"
    assert atmosphere["altitude_km"].tolist() == [0.0, 2.0, 10.0]
    assert atmosphere[PROFILE_THICKNESS_COLUMN].tolist() == [1030.0, 800.0, 270.0]
    assert atmosphere[ATMOSPHERE_REFRACTIVE_INDEX_KEY].tolist() == [2.8e-4, 2.3e-4, 1.0e-4]
    assert math.isclose(telescopes[0]["r"], 200.0, rel_tol=1e-6, abs_tol=1e-12)


# SHA-256 of the file generated from ``_reference_events`` and ``atmospheric_profile`` with
# ``run_number=7``; compressed files are checked after decompression.
REFERENCE_SHA256 = "c70b7edc6d86888018ec7734e095a80f2e6cb70939afd02daeedfaa24b5a7314"


def _reference_events():
    rng = np.random.default_rng(20261025)
    events = []
    for n in (1000, 0, 257):
        bunches = np.zeros(n, dtype=BUNCH_DTYPE)
        for name in ("x", "y", "cx", "cy", "time", "zem", "photons", "wavelength"):
            bunches[name] = rng.uniform(-50.0, 50.0, n)
        bunches["photons"] = rng.uniform(0.0, 1.0, n)
        events.append(bunches)
    return events


@pytest.mark.parametrize("name", PHOTON_FILE_NAMES)
def test_write_photon_bunches_reference_bytes(tmp_test_directory, atmospheric_profile, name):
    path = tmp_test_directory / name
    write_photon_bunches(
        path,
        _reference_events(),
        observation_level_m=2147.0,
        telescope_radius_m=2.0,
        atmospheric_profile=atmospheric_profile,
        run_number=7,
    )
    path = Path(path)
    content = gzip.open(path).read() if name.endswith(".gz") else Path(path).read_bytes()
    assert hashlib.sha256(content).hexdigest() == REFERENCE_SHA256


def _camera_photons():
    return {
        "position_m": np.array([[0.1, 0.2, 16.0]]),
        "direction": np.array([[0.01, 0.02, 0.99975]]),
        "time_ns": np.array([55.0]),
        "weight": np.array([0.7]),
        "wavelength_nm": np.array([385.0]),
    }


def test_camera_bunches_single_mirror_turn_the_frame_about_x():
    bunch = camera_bunches(_camera_photons(), dual_mirror=False, emission_altitude_m=2147.0)[0]
    assert np.allclose((bunch["x"], bunch["y"]), (10.0, -20.0), rtol=1e-6, atol=1e-12)
    assert np.allclose((bunch["cx"], bunch["cy"]), (0.01, -0.02), rtol=1e-6, atol=1e-12)
    assert math.isclose(bunch["time"], 55.0, rel_tol=1e-6, abs_tol=1e-12)
    assert math.isclose(bunch["zem"], 214700.0, rel_tol=1e-6, abs_tol=1e-12)
    assert math.isclose(bunch["photons"], 0.7, rel_tol=1e-6, abs_tol=1e-12)
    assert math.isclose(bunch["wavelength"], 385.0, rel_tol=1e-6, abs_tol=1e-12)
    # The camera rotation does not matter for single-mirror telescopes.
    rotated = camera_bunches(
        _camera_photons(), dual_mirror=False, emission_altitude_m=2147.0, camera_rotation_deg=30.0
    )
    assert np.array_equal(
        rotated, camera_bunches(_camera_photons(), dual_mirror=False, emission_altitude_m=2147.0)
    )


def test_camera_bunches_dual_mirror_in_the_camera_frame():
    bunch = camera_bunches(_camera_photons(), dual_mirror=True, emission_altitude_m=2147.0)[0]
    assert np.allclose((bunch["x"], bunch["y"]), (10.0, 20.0), rtol=1e-6, atol=1e-12)
    assert np.allclose((bunch["cx"], bunch["cy"]), (0.01, 0.02), rtol=1e-6, atol=1e-12)
    # Camera rotated by 90 deg: telescope x is camera -y.
    bunch = camera_bunches(
        _camera_photons(), dual_mirror=True, emission_altitude_m=2147.0, camera_rotation_deg=90.0
    )[0]
    assert np.allclose((bunch["x"], bunch["y"]), (20.0, -10.0), rtol=1e-6, atol=1e-12)
    assert np.allclose((bunch["cx"], bunch["cy"]), (0.02, -0.01), rtol=1e-6, atol=1e-12)


def test_bypass_optics_level():
    assert bypass_optics_level(False) == 1
    assert bypass_optics_level(True) == 2


def _write_reference(destination, atmospheric_profile):
    return write_photon_bunches(
        destination,
        _reference_events(),
        observation_level_m=2147.0,
        telescope_radius_m=2.0,
        atmospheric_profile=atmospheric_profile,
        run_number=7,
    )


def test_write_to_stream_is_byte_identical_to_file(tmp_test_directory, atmospheric_profile):
    path = tmp_test_directory / "photons.iact"
    _write_reference(path, atmospheric_profile)

    buffer = io.BytesIO()
    assert _write_reference(buffer, atmospheric_profile) == 3
    assert not buffer.closed
    assert buffer.getvalue() == Path(path).read_bytes()
    assert hashlib.sha256(buffer.getvalue()).hexdigest() == REFERENCE_SHA256


def test_write_to_stdout_dash(monkeypatch, tmp_test_directory, atmospheric_profile):
    path = tmp_test_directory / "photons.iact"
    _write_reference(path, atmospheric_profile)

    buffer = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(buffer))
    assert _write_reference("-", atmospheric_profile) == 3
    assert buffer.getvalue() == Path(path).read_bytes()


def test_write_to_text_stream_raises(tmp_test_directory, atmospheric_profile):
    with pytest.raises(TypeError, match="binary stream"):
        _write_reference(io.StringIO(), atmospheric_profile)


def test_write_several_telescopes_per_event(tmp_test_directory, atmospheric_profile):
    path = tmp_test_directory / "photons.iact"
    events = [
        PhotonBunchEvent(number=7, blocks=((0, _bunches(3)), (1, _bunches(2)))),
        PhotonBunchEvent(number=9, blocks=((2, _bunches(0)),)),
        [(1, _bunches(4)), (0, _bunches(1))],
    ]
    count = write_photon_bunches(
        path,
        events,
        observation_level_m=2147.0,
        telescope_radius_m=2.0,
        atmospheric_profile=atmospheric_profile,
    )

    assert count == 3
    objects = []
    blocks = []
    for item in EventIOFile(str(path)):
        objects.append(item.header.type)
        if item.header.type == 1202:
            event_number = int(item.parse()["event_number"])
        elif item.header.type == 1204:
            photons = next(iter(item))
            blocks.append((event_number, photons.telescope, photons.parse()[0]))
    # One photon bunch object (1204) per telescope block, after the event header.
    assert objects == [
        1200,
        1201,
        1216,
        1202,
        1203,
        1204,
        1204,
        1209,
        1202,
        1203,
        1204,
        1209,
        1202,
        1203,
        1204,
        1204,
        1209,
        1210,
    ]
    assert [(number, telescope, len(bunches)) for number, telescope, bunches in blocks] == [
        (7, 0, 3),
        (7, 1, 2),
        (9, 2, 0),
        (3, 1, 4),
        (3, 0, 1),
    ]
    assert np.array_equal(blocks[1][2], events[0].blocks[1][1])


def test_write_telescope_id_out_of_range_raises(atmospheric_profile):
    with pytest.raises(ValueError, match="Telescope id must be from 0 to 32767"):
        write_photon_bunches(
            io.BytesIO(),
            [[(40000, _bunches(1))]],
            observation_level_m=2147.0,
            telescope_radius_m=2.0,
            atmospheric_profile=atmospheric_profile,
        )


def test_write_telescope_id_not_integer_raises(atmospheric_profile):
    with pytest.raises(TypeError, match="Telescope id must be an integer"):
        write_photon_bunches(
            io.BytesIO(),
            [[("1", _bunches(1))]],
            observation_level_m=2147.0,
            telescope_radius_m=2.0,
            atmospheric_profile=atmospheric_profile,
        )


def test_write_core_reuses_share_one_shower_header(tmp_test_directory, atmospheric_profile):
    """Consecutive events with one number are core reuses: one shower header, reuse identifiers."""
    events = [
        PhotonBunchEvent(number=number, blocks=((0, _bunches(3)),)) for number in (1, 1, 1, 2, 2)
    ]
    path = Path(tmp_test_directory) / "reuses.iact"

    count = write_photon_bunches(
        path,
        events,
        observation_level_m=2147.0,
        telescope_radius_m=12.5,
        atmospheric_profile=atmospheric_profile,
        reuses=3,
    )

    assert count == 5
    objects = list(EventIOFile(path))
    names = [type(obj).__name__ for obj in objects]
    assert names.count("EventHeader") == 2
    assert names.count("TelescopeData") == 5
    # The identifier of a telescope block is its core-reuse index within the shower.
    reuse_ids = [obj.header.id for obj in objects if type(obj).__name__ == "TelescopeData"]
    assert reuse_ids == [0, 1, 2, 0, 1]


class _FlushCounter(io.BytesIO):
    """Binary stream counting its flushes."""

    def __init__(self):
        super().__init__()
        self.flushes = 0

    def flush(self):
        self.flushes += 1
        super().flush()


def test_write_flushes_each_event_before_the_next_is_read(atmospheric_profile):
    output = _FlushCounter()
    flushes_before_event = []

    def events():
        for number in (1, 2, 3):
            flushes_before_event.append(output.flushes)
            yield PhotonBunchEvent(number=number, blocks=((0, _bunches(2)),))

    write_photon_bunches(
        output,
        events(),
        observation_level_m=2147.0,
        telescope_radius_m=2.0,
        atmospheric_profile=atmospheric_profile,
    )

    # Each event is flushed before the producer is asked for the next one.
    assert flushes_before_event[1] > flushes_before_event[0]
    assert flushes_before_event[2] > flushes_before_event[1]
