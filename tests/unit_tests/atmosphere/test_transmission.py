#!/usr/bin/python3

import math

import astropy.units as u
import pytest
from astropy.table import Table

from simtools.atmosphere.transmission import AtmosphericTransmission


def _table():
    """Two wavelengths; 301 nm lists only the lower altitude (as UV rows do)."""
    table = Table(
        {
            "wavelength": [300.0, 300.0, 301.0] * u.nm,
            "altitude": [3.0, 12.0, 3.0] * u.km,
            "extinction": [0.10, 0.40, 0.20],
        }
    )
    table.meta["observatory_level"] = 2.0 * u.km
    return table


def test_atmospheric_transmission():
    atmosphere = AtmosphericTransmission(_table())
    assert math.isclose(atmosphere.observation_level_m, 2000.0, rel_tol=1e-6, abs_tol=1e-12)
    assert math.isclose(atmosphere.optical_depth(300.0, 2000.0), 0.0, rel_tol=1e-6, abs_tol=1e-12)
    assert math.isclose(
        atmosphere.optical_depth(300.4, 3000.0), 0.10, rel_tol=1e-6, abs_tol=1e-12
    )  # Wavelength below.
    # Linear in log(altitude) between the levels, clamped beyond them.
    middle = 10 ** ((math.log10(3000.0) + math.log10(12000.0)) / 2)
    assert math.isclose(atmosphere.optical_depth(300.0, middle), 0.25, rel_tol=1e-6, abs_tol=1e-12)
    assert math.isclose(
        atmosphere.optical_depth(900.0, 50000.0), 0.20, rel_tol=1e-6, abs_tol=1e-12
    )  # Only to 3 km.
    assert math.isclose(atmosphere.optical_depth(100.0, 3000.0), 0.10, rel_tol=1e-6, abs_tol=1e-12)
    # The depth between emitter and telescope, scaled by the airmass.
    transmission = atmosphere.transmission(300.0, 3000.0, 2000.0, 2.0)
    assert transmission == pytest.approx(math.exp(-0.2))
    assert math.isclose(
        atmosphere.transmission(300.0, 1999.0, 2000.0, 1.0), 0.0, rel_tol=1e-6, abs_tol=1e-12
    )
