#!/usr/bin/python3

import numpy as np
import pytest
from eventio import EventIOFile

from simtools.simtel.photon_bunch_file import (
    BUNCH_DTYPE,
    bypass_optics_level,
    camera_bunches,
    write_photon_bunches,
)


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


@pytest.mark.parametrize("name", ["photons.iact", "photons.iact.gz"])
def test_write_photon_bunches_reads_back(tmp_test_directory, name):
    path = tmp_test_directory / name
    events = [_bunches(3), _bunches(2, offset=10.0)]

    count = write_photon_bunches(
        path, events, observation_level_m=2147.0, telescope_radius_m=2.0, run_number=7
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
        elif item.header.type == 1204:
            photons = next(iter(item))
            bunches.append(photons.parse()[0])
            assert photons.n_bunches == len(bunches[-1])
            assert photons.n_photons == pytest.approx(bunches[-1]["photons"].sum())
    assert objects == [1200, 1201, 1202, 1203, 1204, 1209, 1202, 1203, 1204, 1209, 1210]
    for written, read in zip(events, bunches):
        assert np.array_equal(written, read)
    # CORSIKA words (1-based): run number 2, observation level 6 (run) or 48 (event).
    assert run_header[0] == b"RUNH"
    assert run_header[1][1] == 7
    assert run_header[1][5] == pytest.approx(214700.0)
    assert event_header[0] == b"EVTH"
    assert event_header[1][1] == 2
    assert event_header[1][47] == pytest.approx(214700.0)
    # Word 77: atmospheric profile 1 (atmprof1.dat) times 1024 plus option bits.
    assert int(event_header[1][76]) >> 10 == 1
    assert telescopes[0]["r"] == pytest.approx(200.0)


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
