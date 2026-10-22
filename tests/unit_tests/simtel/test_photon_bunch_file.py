#!/usr/bin/python3

import numpy as np
import pytest
from eventio import EventIOFile

from simtools.simtel.photon_bunch_file import BUNCH_DTYPE, write_photon_bunches


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
