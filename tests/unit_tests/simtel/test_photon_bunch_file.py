#!/usr/bin/python3

import gzip
import hashlib
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table
from eventio import EventIOFile

from simtools.simtel.photon_bunch_file import (
    BUNCH_DTYPE,
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
    assert run_header[1][5] == pytest.approx(214700.0)
    assert event_header[0] == b"EVTH"
    assert event_header[1][1] == 2
    assert event_header[1][47] == pytest.approx(214700.0)
    # Word 77: the number of the profile in the file times 1024 plus option bits.
    assert int(event_header[1][76]) >> 10 == atmosphere["id"]
    assert atmosphere["obslevel"] == pytest.approx(214700.0)
    assert atmosphere["name"] == b"atmospheric_profile"
    assert atmosphere["altitude_km"].tolist() == [0.0, 2.0, 10.0]
    assert atmosphere[PROFILE_THICKNESS_COLUMN].tolist() == [1030.0, 800.0, 270.0]
    assert atmosphere[ATMOSPHERE_REFRACTIVE_INDEX_KEY].tolist() == [2.8e-4, 2.3e-4, 1.0e-4]
    assert telescopes[0]["r"] == pytest.approx(200.0)


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
    content = gzip.open(path).read() if name.endswith(".gz") else path.read_bytes()
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
    assert (bunch["x"], bunch["y"]) == pytest.approx((10.0, -20.0))
    assert (bunch["cx"], bunch["cy"]) == pytest.approx((0.01, -0.02))
    assert bunch["time"] == pytest.approx(55.0)
    assert bunch["zem"] == pytest.approx(214700.0)
    assert bunch["photons"] == pytest.approx(0.7)
    assert bunch["wavelength"] == pytest.approx(385.0)
    # The camera rotation does not matter for single-mirror telescopes.
    rotated = camera_bunches(
        _camera_photons(), dual_mirror=False, emission_altitude_m=2147.0, camera_rotation_deg=30.0
    )
    assert np.array_equal(
        rotated, camera_bunches(_camera_photons(), dual_mirror=False, emission_altitude_m=2147.0)
    )


def test_camera_bunches_dual_mirror_in_the_camera_frame():
    bunch = camera_bunches(_camera_photons(), dual_mirror=True, emission_altitude_m=2147.0)[0]
    assert (bunch["x"], bunch["y"]) == pytest.approx((10.0, 20.0))
    assert (bunch["cx"], bunch["cy"]) == pytest.approx((0.01, 0.02))
    # Camera rotated by 90 deg: telescope x is camera -y.
    bunch = camera_bunches(
        _camera_photons(), dual_mirror=True, emission_altitude_m=2147.0, camera_rotation_deg=90.0
    )[0]
    assert (bunch["x"], bunch["y"]) == pytest.approx((20.0, -10.0))
    assert (bunch["cx"], bunch["cy"]) == pytest.approx((0.02, -0.01))


def test_bypass_optics_level():
    assert bypass_optics_level(False) == 1
    assert bypass_optics_level(True) == 2
