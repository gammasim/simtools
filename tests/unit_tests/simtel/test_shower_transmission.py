"""Tests of the sim_telarray atmospheric-transmission lookup used by shower routing."""

import math
from pathlib import Path

import pytest

from simtools.simtel.shower_transmission import SimtelTransmission, transmission_ratio


@pytest.fixture
def transmission_table(tmp_test_directory):
    path = Path(tmp_test_directory) / "transmission.dat"
    path.write_text(
        "# H2= 2.147, H1= 3 10 30\n400 0.1 0.3 0.5\n401 0.2 0.4 0.6\n",
        encoding="utf-8",
    )
    return SimtelTransmission(path, 2147.0)


def test_transmission_ratio_identity(transmission_table):
    ratio = transmission_ratio(transmission_table, 400, 9000, 1.2, 1.2, 34, 34, 28, 2147)
    assert ratio == pytest.approx(1.0)


@pytest.mark.parametrize("zenith", [20, 40, 60])
def test_transmission_ratio_matches_direct_lookup(transmission_table, zenith):
    ground_airmass = 1 / math.cos(math.radians(zenith))
    camera_airmass = 1.001
    native = transmission_table.transmission(400, 9000, 34, ground_airmass, 28, 2147)
    bypass = transmission_table.transmission(400, 9000, 0, camera_airmass, 28, 2147)
    ratio = transmission_ratio(
        transmission_table, 400, 9000, ground_airmass, camera_airmass, 34, 0, 28, 2147
    )
    assert ratio == pytest.approx(native / bypass)


def test_transmission_ratio_zero_bypass(transmission_table):
    assert transmission_ratio(transmission_table, 400, 1000, 1, 1, 34, 0, 28, 2147) is None


def test_transmission_resets_depths_for_invalid_telescope_level(transmission_table):
    result = transmission_table.transmission(400, 9000, -10, 1.5, 28, 2147)
    expected = transmission_table.atmospheric_transmission2(400, 900000, 1.5, 0, 0)
    assert result == pytest.approx(expected)
