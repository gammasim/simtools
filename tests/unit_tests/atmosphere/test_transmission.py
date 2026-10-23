#!/usr/bin/python3

import math

import pytest

from simtools.atmosphere.transmission import AtmosphericTransmission


def test_atmospheric_transmission(tmp_test_directory):
    table = tmp_test_directory / "atm_trans.dat"
    table.write_text(
        "# Optical depth between altitudes\n"
        "# H2= 2.000, H1= 3.000 12.000\n"
        "300 0.10 0.40\n"
        "301 0.20 0.80\n",
        encoding="utf-8",
    )
    atmosphere = AtmosphericTransmission(table)
    assert atmosphere.observation_level_m == pytest.approx(2000.0)
    assert atmosphere.optical_depth(300.0, 2000.0) == pytest.approx(0.0)
    assert atmosphere.optical_depth(300.4, 3000.0) == pytest.approx(0.10)  # Integer wavelength.
    # Linear in log10(altitude) between the levels, clamped beyond them.
    middle = 10 ** ((math.log10(3000.0) + math.log10(12000.0)) / 2)
    assert atmosphere.optical_depth(301.0, middle) == pytest.approx(0.50)
    assert atmosphere.optical_depth(900.0, 50000.0) == pytest.approx(0.80)
    # The depth between emitter and telescope, scaled by the airmass.
    transmission = atmosphere.transmission(300.0, 3000.0, 2000.0, 2.0)
    assert transmission == pytest.approx(math.exp(-0.2))
    assert atmosphere.transmission(300.0, 1999.0, 2000.0, 1.0) == pytest.approx(0.0)
