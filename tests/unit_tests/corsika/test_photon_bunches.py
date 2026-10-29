import struct
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table

from simtools.corsika.photon_bunches import read_photon_bunches
from simtools.simtel.photon_bunch_file import BUNCH_DTYPE, write_photon_bunches

BUNCH_FIELDS = {
    "x": "x",
    "y": "y",
    "cx": "cx",
    "cy": "cy",
    "time_ns": "time",
    "zem": "zem",
    "size": "photons",
    "wavelength_nm": "wavelength",
}
PHOTON_OBJECT_TYPE = 1205
COMPACT_FORMAT_VERSION = 1000
OBJECT_TYPE_BITS = 20


def _bunches(n, offset=0.0, wavelength=385.0):
    bunches = np.zeros(n, dtype=BUNCH_DTYPE)
    bunches["x"] = np.arange(n) + offset
    bunches["y"] = -np.arange(n)
    bunches["cx"], bunches["cy"] = 0.01, -0.02
    bunches["time"] = 55.0
    bunches["zem"] = 216350.0
    bunches["photons"] = 0.5
    bunches["wavelength"] = wavelength
    return bunches


@pytest.fixture
def atmospheric_profile():
    return Table(
        {
            "altitude": [0.0, 2.0, 10.0] * u.km,
            "density": [1.2e-3, 1.0e-3, 4.0e-4] * u.g / u.cm**3,
            "thickness": [1030.0, 800.0, 270.0] * u.g / u.cm**2,
            "refractive_index": [2.8e-4, 2.3e-4, 1.0e-4],
        }
    )


def _write(path, events, atmospheric_profile):
    return write_photon_bunches(
        path,
        events,
        observation_level_m=2147.0,
        telescope_radius_m=2.0,
        atmospheric_profile=atmospheric_profile,
    )


def test_round_trip(tmp_test_directory, atmospheric_profile):
    path = tmp_test_directory / "photons.iact"
    events = [_bunches(3), _bunches(2, offset=10.0)]
    _write(path, events, atmospheric_profile)

    telescopes = list(read_photon_bunches(path))

    assert len(telescopes) == len(events)
    for event_number, (telescope, written) in enumerate(zip(telescopes, events, strict=True), 1):
        assert telescope["event_number"] == event_number
        assert telescope["telescope_id"] == 0
        assert telescope["skipped_bunches"] == 0
        assert telescope["wavelength_min_nm"] == pytest.approx(200.0)
        assert telescope["wavelength_max_nm"] == pytest.approx(1000.0)
        for key, field in BUNCH_FIELDS.items():
            assert telescope[key].dtype == np.float32
            np.testing.assert_array_equal(telescope[key], written[field])
        assert telescope["bunch_id"].dtype == np.uint64


def test_skipped_wavelengths_are_counted(tmp_test_directory, atmospheric_profile):
    path = tmp_test_directory / "photons.iact"
    event = _bunches(5)
    event["wavelength"] = [400.0, 1.0, 9500.0, 9000.0, 300.0]
    _write(path, [event], atmospheric_profile)

    (telescope,) = read_photon_bunches(path)

    assert telescope["skipped_bunches"] == 3
    np.testing.assert_array_equal(telescope["wavelength_nm"], [400.0, 300.0])
    np.testing.assert_array_equal(telescope["x"], event["x"][[0, 4]])
    expected_ids = (np.uint64(1) << np.uint64(32)) | np.array([0, 4], dtype=np.uint64)
    np.testing.assert_array_equal(telescope["bunch_id"], expected_ids)


def test_negative_wavelength_raises(tmp_test_directory, atmospheric_profile):
    path = tmp_test_directory / "photons.iact"
    _write(path, [_bunches(2), _bunches(2, wavelength=-1.0)], atmospheric_profile)

    with pytest.raises(ValueError, match="Photo-electron bunches"):
        list(read_photon_bunches(path))


def test_bunch_ids_are_unique_and_encode_event_and_index(tmp_test_directory, atmospheric_profile):
    path = tmp_test_directory / "photons.iact"
    events = [_bunches(3), _bunches(4), _bunches(2)]
    _write(path, events, atmospheric_profile)

    telescopes = list(read_photon_bunches(path))

    all_ids = np.concatenate([telescope["bunch_id"] for telescope in telescopes])
    assert len(np.unique(all_ids)) == sum(len(event) for event in events)
    for telescope, written in zip(telescopes, events, strict=True):
        expected = (np.uint64(telescope["event_number"]) << np.uint64(32)) | np.arange(
            len(written), dtype=np.uint64
        )
        np.testing.assert_array_equal(telescope["bunch_id"], expected)


def test_compact_format_raises(tmp_test_directory, atmospheric_profile):
    path = Path(tmp_test_directory) / "photons.iact"
    _write(path, [_bunches(2)], atmospheric_profile)
    data = bytearray(path.read_bytes())
    header = struct.pack("<Ii", PHOTON_OBJECT_TYPE, 0)
    position = data.index(header)
    struct.pack_into(
        "<I",
        data,
        position,
        PHOTON_OBJECT_TYPE | (COMPACT_FORMAT_VERSION << OBJECT_TYPE_BITS),
    )
    path.write_bytes(bytes(data))

    with pytest.raises(NotImplementedError, match=path.name):
        list(read_photon_bunches(path))
