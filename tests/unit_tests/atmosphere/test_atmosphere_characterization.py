"""Tests for atmospheric characterization reports."""

import json
import math
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.table import QTable
from astropy.tests.helper import assert_quantity_allclose

from simtools.atmosphere.atmosphere_characterization import (
    _TOP_OF_ATMOSPHERE,
    _plot_profiles,
    _plot_transmission,
    _prepare_atmosphere,
    _profile_difference,
    _profile_relative_difference,
    _profile_samples,
    _transmission_at_altitude,
    _transmission_levels,
    _transmission_relative_difference,
    _transmission_slice,
    characterize_atmospheres,
)
from simtools.visualization.matplotlib_backend import pyplot as plt

_DENSITY = "density"
_HEIGHT = "height"
_PROFILE_VERSION = "profile_version"
_SITE = "site"
_VERSION = "1.0.0"
_REFERENCE = "reference"
_TRANSMISSION_DIFFERENCE = "transmission_difference"
_RELATIVE_DIFFERENCE_LIMIT = 35.0
_SUMMARY_FILE = "atmosphere_summary.json"
_PROFILE_PLOT = "atmosphere_profiles.png"
_TRANSMISSION_PLOT = "atmosphere_transmission.png"
_WAVELENGTH = "wavelength"
_EXTINCTION = "extinction"
_ATMOSPHERES = "atmospheres"

_PROFILE = "profile"
_TRANSMISSION = "transmission"
_ALTITUDE = "altitude"
_LABEL = "label"
_LEVELS = _transmission_levels(2 * u.km)


@pytest.fixture
def atmospheres():
    """Create two small model tables with overlapping wavelengths."""
    profile = QTable(
        {
            _ALTITUDE: [5, 10, 15] * u.km,
            _DENSITY: [0.7, 0.4, 0.2] * u.g / u.cm**3,
            "thickness": [500, 300, 100] * u.g / u.cm**2,
            "refractive_index": [0.0002, 0.0001, 0.00005],
            "temperature": [280, 270, 260] * u.K,
            "pressure": [800, 500, 300] * u.mbar,
            "water_vapor_partial_pressure": [0.01, 0.005, 0.001],
        }
    )
    transmission = QTable(
        {
            _ALTITUDE: [10, 10, 100, 100, 20, 20, 5, 5, 1.5, 1.5, 2, 2] * u.km,
            _WAVELENGTH: [300, 400] * 6 * u.nm,
            _EXTINCTION: [0, np.log(2)] * 6,
        },
        meta={"observatory_level": 2 * u.km},
    )
    return [
        {_LABEL: _REFERENCE, _PROFILE: profile, _TRANSMISSION: transmission, _SITE: "North"},
        {
            _LABEL: "candidate",
            _PROFILE: profile.copy(copy_data=True),
            _TRANSMISSION: transmission.copy(copy_data=True),
        },
    ]


def test_characterize_writes_plots_and_summary(atmospheres, tmp_test_directory):
    tmp_test_directory = Path(tmp_test_directory)
    atmospheres[1][_TRANSMISSION][_EXTINCTION][1] = 0
    result = characterize_atmospheres(atmospheres, tmp_test_directory)
    assert result[_REFERENCE] == _REFERENCE
    candidate = result[_ATMOSPHERES][1][_TRANSMISSION_DIFFERENCE]
    assert candidate["common_wavelength_count"] == 2
    assert math.isclose(candidate["max_absolute_difference"], 0.5)
    assert math.isclose(result[_ATMOSPHERES][0]["observation_level_km"], 2.0)
    for name in (
        _PROFILE_PLOT,
        _TRANSMISSION_PLOT,
        _SUMMARY_FILE,
    ):
        assert (tmp_test_directory / name).is_file()
    assert json.loads((tmp_test_directory / _SUMMARY_FILE).read_text()) == result


def test_characterize_writes_candidate_only_plots(atmospheres, tmp_test_directory):
    result = characterize_atmospheres(atmospheres[:1], tmp_test_directory)

    assert result[_REFERENCE] == _REFERENCE
    assert result[_ATMOSPHERES][0][_TRANSMISSION_DIFFERENCE]["common_wavelength_count"] == 2
    assert (Path(tmp_test_directory) / _PROFILE_PLOT).is_file()
    assert (Path(tmp_test_directory) / _TRANSMISSION_PLOT).is_file()


def test_characterize_validates_inputs(atmospheres, tmp_test_directory):
    with pytest.raises(ValueError, match="At least one atmosphere"):
        characterize_atmospheres([], tmp_test_directory)
    atmospheres[1][_LABEL] = atmospheres[0][_LABEL]
    with pytest.raises(ValueError, match="labels must be unique"):
        characterize_atmospheres(atmospheres, tmp_test_directory)
    with pytest.raises(ValueError, match="limit must be positive"):
        characterize_atmospheres(atmospheres[:1], tmp_test_directory, 0)


def test_characterize_disjoint_wavelengths(atmospheres, tmp_test_directory):
    atmospheres[1][_TRANSMISSION][_WAVELENGTH] = [500, 600] * 6 * u.nm
    result = characterize_atmospheres(atmospheres, tmp_test_directory)
    assert result[_ATMOSPHERES][1][_TRANSMISSION_DIFFERENCE] is None


def test_profile_differences_interpolate_only_over_overlap():
    reference = {_HEIGHT: np.array([0, 2, 4]), _PROFILE: {_DENSITY: np.array([4, 2, 0])}}
    candidate = {_HEIGHT: np.array([1, 3]), _PROFILE: {_DENSITY: np.array([4, 2])}}
    altitude, difference = _profile_difference(reference, candidate, _DENSITY)
    assert np.array_equal(altitude, [1, 2, 3])
    assert np.allclose(difference, [1, 1, 1])
    candidate[_HEIGHT] = np.array([5, 6])
    altitude, difference = _profile_difference(reference, candidate, _DENSITY)
    assert altitude.size == difference.size == 0


def test_profile_relative_differences_are_percentages():
    reference = {_HEIGHT: np.array([0, 2]), _PROFILE: {_DENSITY: np.array([2, 4])}}
    candidate = {_HEIGHT: np.array([0, 2]), _PROFILE: {_DENSITY: np.array([3, 2])}}
    altitude, percentage = _profile_relative_difference(reference, candidate, _DENSITY)
    assert np.array_equal(altitude, [0, 2])
    assert np.allclose(percentage, [50, -50])


def test_transmission_relative_differences_are_percentages():
    reference = {_WAVELENGTH: np.array([300, 400]), _TRANSMISSION: np.array([0.5, 0.0])}
    candidate = {_WAVELENGTH: np.array([300, 400]), _TRANSMISSION: np.array([0.75, 0.5])}
    wavelength, percentage = _transmission_relative_difference(reference, candidate)
    assert np.array_equal(wavelength, [300, 400])
    assert np.isclose(percentage[0], 50)
    assert np.isnan(percentage[1])


def test_profile_figure_has_altitude_and_reference_differences(
    atmospheres, mocker, tmp_test_directory
):
    first, second = atmospheres
    first[_PROFILE_VERSION] = _VERSION
    second[_SITE] = "South"
    second[_PROFILE_VERSION] = _VERSION
    second[_PROFILE][_DENSITY] *= 2
    prepared = [_prepare_atmosphere(item, _LEVELS) for item in atmospheres]
    save = mocker.patch("simtools.atmosphere.atmosphere_characterization.visualize.save_figure")
    _plot_profiles(prepared, tmp_test_directory, _RELATIVE_DIFFERENCE_LIMIT)
    figure = save.call_args.args[0]
    assert len(figure.axes) == 18
    assert figure.axes[6].get_ylabel().startswith("Difference")
    assert figure.axes[0].get_ylabel().startswith("Density")
    assert figure.axes[0].get_yscale() == "log"
    assert figure.axes[12].get_ylabel() == "Relative difference [%]"
    assert figure.axes[12].get_xlabel() == "Altitude a.s.l. [km]"
    assert figure.axes[12].get_ylim() == (-35.0, 35.0)
    assert figure.axes[0].get_legend_handles_labels()[1] == [
        "North profile 1.0.0",
        "South profile 1.0.0",
    ]
    assert np.allclose(figure.axes[6].lines[1].get_ydata(), [0.7, 0.4, 0.2])
    assert np.allclose(figure.axes[6].lines[0].get_ydata(), 0)
    plt.close(figure)


def test_transmission_figure_has_requested_levels(atmospheres, mocker, tmp_test_directory):
    prepared = [_prepare_atmosphere(item, _LEVELS) for item in atmospheres]
    save = mocker.patch("simtools.atmosphere.atmosphere_characterization.visualize.save_figure")
    _plot_transmission(prepared, tmp_test_directory, _RELATIVE_DIFFERENCE_LIMIT)
    figure = save.call_args.args[0]
    assert len(figure.axes) == 12
    assert figure.axes[0].get_title() == _TOP_OF_ATMOSPHERE
    assert figure.axes[8].get_title() == "3.5 km a.s.l."
    assert figure.axes[10].get_title() == "2.5 km a.s.l."
    assert figure.axes[10].get_xlabel() == "Wavelength [nm]"
    assert figure.axes[11].get_ylabel() == "Relative difference [%]"
    assert figure.axes[11].get_ylim() == (-35.0, 35.0)
    plt.close(figure)


def test_transmission_levels_use_highest_observation_level():
    labels = [label for label, _ in _transmission_levels(2.2 * u.km)]
    assert labels == [
        _TOP_OF_ATMOSPHERE,
        "20 km a.s.l.",
        "10 km a.s.l.",
        "5 km a.s.l.",
        "3.7 km a.s.l.",
        "2.7 km a.s.l.",
    ]


def test_prepare_atmosphere_requires_observatory_level(atmospheres):
    atmospheres[0][_TRANSMISSION].meta.clear()
    with pytest.raises(ValueError, match="must define observatory_level"):
        _prepare_atmosphere(atmospheres[0], _LEVELS)


_UNIQUE = "must be unique"
_FINITE_ALTITUDE = "altitude must be finite"


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
    assert math.isclose(result[_DENSITY], 3.0)
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


def test_transmission_at_altitude_extrapolates_and_clips(transmission):
    _, values = _transmission_at_altitude(transmission, 12.5 * u.km)
    assert np.allclose(values, [1, np.exp(-1.5 * np.log(2))])
    _, values = _transmission_at_altitude(transmission, 2.5 * u.km)
    assert np.allclose(values, [np.exp(-1.5), 1])
    _, values = _transmission_at_altitude(transmission, 20 * u.km)
    assert np.allclose(values[0], 1)


def test_transmission_at_altitude_skips_single_altitude_wavelengths(transmission):
    transmission.add_row([5 * u.km, 500 * u.nm, 0.1])
    wavelength, _ = _transmission_at_altitude(transmission, 7.5 * u.km)
    assert_quantity_allclose(wavelength, [300, 400] * u.nm)
    with pytest.raises(ValueError, match="at least two altitudes"):
        _transmission_at_altitude(transmission[-1:], 7.5 * u.km)


def test_transmission_at_altitude_rejects_duplicate_altitudes(transmission):
    transmission[_ALTITUDE][0] = 5 * u.km
    transmission[_WAVELENGTH][0] = 300 * u.nm
    with pytest.raises(ValueError, match="once per altitude"):
        _transmission_at_altitude(transmission, 7.5 * u.km)


def test_transmission_rejects_nonfinite_axes(transmission):
    transmission[_WAVELENGTH][0] = np.nan * u.nm
    with pytest.raises(ValueError, match="altitude and wavelength must be finite"):
        _transmission_at_altitude(transmission, 7.5 * u.km)
