"""Tests for internal atmospheric characterization calculations."""

import astropy.units as u
import numpy as np
import pytest
from astropy.table import QTable
from astropy.tests.helper import assert_quantity_allclose

from simtools.atmosphere.atmosphere_characterization_helpers import (
    _profile_samples,
    _transmission_at_altitude,
    _transmission_slice,
)

_UNIQUE = "must be unique"
_FINITE_ALTITUDE = "altitude must be finite"

_ALTITUDE = "altitude"
_DENSITY = "density"
_WAVELENGTH = "wavelength"
_EXTINCTION = "extinction"


@pytest.fixture
def profile():
    """Return a small profile in descending altitude order."""
    return QTable(
        {
            _ALTITUDE: [2, 0] * u.km,
            _DENSITY: [2.0, 4.0],
            "thickness": [3.0, 7.0],
            "refractive_index": [1e-4, 3e-4],
            "temperature": [280, 290] * u.K,
            "pressure": [800, 900] * u.mbar,
            "water_vapor_partial_pressure": [0.01, 0.02],
        }
    )


@pytest.fixture
def transmission():
    """Return optical depths at two altitudes."""
    return QTable(
        {
            _ALTITUDE: [10, 10, 5, 5] * u.km,
            _WAVELENGTH: [400, 300, 300, 400] * u.nm,
            _EXTINCTION: [np.log(2), 0, 1, 0],
        }
    )


def test_profile_samples_interpolate_and_reject_extrapolation(profile):
    result = _profile_samples(profile, 1000 * u.m)
    assert result[_DENSITY] == pytest.approx(3)
    with pytest.raises(ValueError, match="outside the profile range"):
        _profile_samples(profile, 3 * u.km)


def test_profile_invalid_data(profile):
    profile[_ALTITUDE] = [0, 0] * u.km
    with pytest.raises(ValueError, match=_UNIQUE):
        _profile_samples(profile, 0 * u.km)
    profile[_ALTITUDE] = [0, 1] * u.km
    profile[_DENSITY][0] = np.nan
    with pytest.raises(ValueError, match="contains non-finite"):
        _profile_samples(profile, 0 * u.km)


def test_transmission_slice_uses_exact_height_and_sorts(transmission):
    wavelength, values = _transmission_slice(transmission, 10 * u.km)
    assert_quantity_allclose(wavelength, [300, 400] * u.nm)
    assert np.allclose(values, [1, 0.5])
    with pytest.raises(ValueError, match="not tabulated"):
        _transmission_slice(transmission, 6 * u.km)


def test_transmission_at_altitude_interpolates_and_extrapolates(transmission):
    wavelength, values = _transmission_at_altitude(transmission, 7.5 * u.km)
    assert_quantity_allclose(wavelength, [300, 400] * u.nm)
    assert np.allclose(values, np.exp([-0.5, -0.5 * np.log(2)]))


def test_transmission_rejects_invalid_depth_and_duplicate_wavelength(transmission):
    transmission[_EXTINCTION][0] = -1
    with pytest.raises(ValueError, match="must be nonnegative"):
        _transmission_slice(transmission, 10 * u.km)
    transmission[_EXTINCTION][0] = 0
    transmission[_WAVELENGTH][0] = 300 * u.nm
    with pytest.raises(ValueError, match=_UNIQUE):
        _transmission_slice(transmission, 10 * u.km)


def test_profile_rejects_missing_units_and_invalid_altitudes(profile):
    profile[_ALTITUDE] = [0, 1]
    with pytest.raises(ValueError, match="requires units"):
        _profile_samples(profile, 0 * u.km)
    profile[_ALTITUDE] = [0, 1] * u.km
    with pytest.raises(ValueError, match=_FINITE_ALTITUDE):
        _profile_samples(profile, np.nan * u.km)
    profile[_ALTITUDE] = [0, np.nan] * u.km
    with pytest.raises(ValueError, match="at least two finite rows"):
        _profile_samples(profile, 0 * u.km)


def test_transmission_rejects_nonfinite_data(transmission):
    with pytest.raises(ValueError, match=_FINITE_ALTITUDE):
        _transmission_slice(transmission, np.nan * u.km)
    transmission[_EXTINCTION][0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        _transmission_slice(transmission, 10 * u.km)
