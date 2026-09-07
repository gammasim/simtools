"""Tests for the in-process camera-efficiency calculator."""

from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table

from simtools.camera.camera_efficiency_calculator import (
    CameraEfficiencyCalculator,
    _atmospheric_transmission,
    _emission_altitude,
    _interpolate,
    _parameter_table,
    _spectral_curve,
)


def test_interpolate_clips_and_sorts_support_points():
    result = _interpolate([10.0, 0.0], [2.0, 1.0], [-1.0, 5.0, 20.0])
    np.testing.assert_allclose(result, [1.0, 1.5, 2.0])


def test_parameter_table_reads_exported_model_file(mocker, tmp_test_directory):
    class Model:
        config_file_directory = Path(tmp_test_directory)

        @staticmethod
        def get_parameter_value(_):
            return "quantum_efficiency.ecsv"

    expected = Table()
    read_table = mocker.patch(
        "simtools.camera.camera_efficiency_calculator.read_simtel_table", return_value=expected
    )

    assert _parameter_table(Model(), "quantum_efficiency") is expected

    read_table.assert_called_once_with(
        "quantum_efficiency", tmp_test_directory / "quantum_efficiency.ecsv"
    )


def test_spectral_curve_averages_angle_dependent_table():
    table = Table(
        {
            "wavelength": [400.0, 400.0, 500.0, 500.0] * u.nm,
            "angle": [0.0, 10.0, 0.0, 10.0] * u.deg,
            "efficiency": [0.8, 0.4, 0.6, 0.2],
        }
    )

    class Model:
        def get_parameter_table(self, name):
            assert name == "incidence"
            return Table(
                {
                    "incidence_angle": [0.0, 10.0] * u.deg,
                    "fraction": [0.25, 0.75],
                }
            )

    result = _spectral_curve(
        table,
        np.array([400.0, 500.0]),
        model=Model(),
        weighting_parameter="incidence",
    )
    np.testing.assert_allclose(result, [0.5, 0.3])


def test_spectral_curve_rejects_an_incomplete_angle_grid():
    table = Table(
        {
            "wavelength": [400.0, 400.0, 500.0] * u.nm,
            "angle": [0.0, 10.0, 0.0] * u.deg,
            "efficiency": [0.8, 0.4, 0.6],
        }
    )

    with pytest.raises(ValueError, match="every incidence angle"):
        _spectral_curve(table, np.array([400.0, 500.0]))


def test_atmospheric_transmission_uses_log_altitude():
    table = Table(
        {
            "wavelength": [400.0, 400.0] * u.nm,
            "altitude": [1.0, 10.0] * u.km,
            "extinction": [2.0, 0.0],
        }
    )
    result = _atmospheric_transmission(table, np.array([400.0]), 10.0**0.5, 2.0)
    np.testing.assert_allclose(result, [np.exp(-2.0)], rtol=1e-12)


def test_emission_altitude_scales_xmax_by_airmass():
    profile = Table(
        {
            "altitude": [0.0, 10.0] * u.km,
            "thickness": [1000.0, 0.0] * (u.g / u.cm**2),
        }
    )
    assert _emission_altitude(profile, 500.0, 2.0) == pytest.approx(7.5)


def test_dual_mirror_reflectivity_uses_both_incidence_distributions():
    class Camera:
        def get_pixel_shape(self):
            return 1

        def get_pixel_diameter(self):
            return 0.1

    class Model:
        def __init__(self, tables, values, units=None):
            self.tables = tables
            self.values = values
            self.units = units or {}
            self.camera = Camera()

        def get_parameter_table(self, name):
            return self.tables[name]

        def get_parameter_value(self, name):
            return self.values[name]

        def get_parameter_value_with_unit(self, name):
            return self.units[name]

        def get_telescope_effective_focal_length(self, *_args):
            return 10.0

    wavelength_angle_curve = Table(
        {
            "wavelength": [200.0, 200.0, 1000.0, 1000.0] * u.nm,
            "angle": [0.0, 10.0, 0.0, 10.0] * u.deg,
            "reflectivity": [0.8, 0.4, 0.8, 0.4],
        }
    )
    spectrum = Table({"wavelength": [200.0, 1000.0] * u.nm, "efficiency": [0.5, 0.5]})
    atmosphere = Table(
        {
            "wavelength": [200.0, 200.0, 1000.0, 1000.0] * u.nm,
            "altitude": [2.0, 120.0, 2.0, 120.0] * u.km,
            "extinction": [1.0, 0.0, 1.0, 0.0],
        }
    )
    profile = Table(
        {
            "altitude": [0.0, 10.0] * u.km,
            "thickness": [1000.0, 0.0] * (u.g / u.cm**2),
        }
    )
    tables = {
        "quantum_efficiency": spectrum,
        "mirror_reflectivity": wavelength_angle_curve,
        "primary_mirror_incidence_angle": Table(
            {"incidence_angle": [0.0, 10.0] * u.deg, "fraction": [1.0, 0.0]}
        ),
        "secondary_mirror_incidence_angle": Table(
            {"incidence_angle": [0.0, 10.0] * u.deg, "fraction": [0.0, 1.0]}
        ),
        "camera_filter": Table({"wavelength": [200.0, 1000.0] * u.nm, "transmission": [1.0, 1.0]}),
        "lightguide_efficiency_vs_incidence_angle": Table(
            {"angle": [0.0, 20.0] * u.deg, "efficiency": [1.0, 1.0]}
        ),
        "lightguide_efficiency_vs_wavelength": spectrum,
        "fake_mirror_list": Table(
            {
                "mirror_x": [1.0] * u.cm,
                "mirror_y": [0.0] * u.cm,
                "mirror_diameter": [1.0] * u.cm,
                "shape_type": [0.0],
                "mirror_z": [0.0] * u.cm,
            }
        ),
        "atmospheric_transmission": atmosphere,
        "atmospheric_profile": profile,
        "nsb_reference_spectrum": Table(
            {"wavelength": [200.0, 1000.0] * u.nm, "differential_photon_rate": [1.0, 1.0]}
        ),
    }
    telescope = Model(
        tables,
        {
            "mirror_class": 2,
            "camera_transmission": 1.0,
            "telescope_transmission": [0.9],
            "parabolic_dish": False,
        },
        {"primary_mirror_diameter": 10.0 * u.m},
    )
    site = Model(tables, {}, {})

    result = CameraEfficiencyCalculator(telescope, site).calculate()
    assert result["ref"][0] == pytest.approx(0.8 * 0.4)


def test_calculator_returns_camera_efficiency_table():
    class Camera:
        def get_pixel_shape(self):
            return 1

        def get_pixel_diameter(self):
            return 0.1

    class Model:
        def __init__(self, tables, values, units=None):
            self.tables = tables
            self.values = values
            self.units = units or {}
            self.camera = Camera()
            self.config_file_directory = None

        def get_parameter_table(self, name):
            return self.tables[name]

        def get_parameter_value(self, name):
            return self.values[name]

        def get_parameter_value_with_unit(self, name):
            return self.units[name]

        def get_telescope_effective_focal_length(self, *_args):
            return 10.0

    spectrum = Table({"wavelength": [200.0, 1000.0] * u.nm, "efficiency": [0.5, 0.5]})
    nsb_spectrum = Table(
        {"wavelength": [200.0, 1000.0] * u.nm, "differential_photon_rate": [1.0, 1.0]}
    )
    atmosphere = Table(
        {
            "wavelength": [200.0, 200.0, 1000.0, 1000.0] * u.nm,
            "altitude": [2.0, 120.0, 2.0, 120.0] * u.km,
            "extinction": [1.0, 0.0, 1.0, 0.0],
        }
    )
    profile = Table(
        {
            "altitude": [0.0, 10.0] * u.km,
            "thickness": [1000.0, 0.0] * (u.g / u.cm**2),
        }
    )
    tables = {
        "quantum_efficiency": spectrum,
        "mirror_reflectivity": spectrum.copy(),
        "camera_filter": Table({"wavelength": [200.0, 1000.0] * u.nm, "transmission": [1.0, 1.0]}),
        "lightguide_efficiency_vs_incidence_angle": Table(
            {"angle": [0.0, 20.0] * u.deg, "efficiency": [0.8, 0.8]}
        ),
        "lightguide_efficiency_vs_wavelength": spectrum.copy(),
        "mirror_list": Table(
            {
                "mirror_x": [1.0] * u.cm,
                "mirror_y": [0.0] * u.cm,
                "mirror_diameter": [1.0] * u.cm,
                "shape_type": [0.0],
                "mirror_z": [0.0] * u.cm,
            }
        ),
        "atmospheric_transmission": atmosphere,
        "atmospheric_profile": profile,
        "nsb_reference_spectrum": nsb_spectrum,
    }
    telescope = Model(
        tables,
        {
            "mirror_class": 1,
            "camera_transmission": 1.0,
            "telescope_transmission": [0.9],
            "parabolic_dish": False,
        },
        {"dish_shape_length": 10.0 * u.m},
    )
    site = Model(
        tables,
        {},
        {"corsika_observation_level": 2.0 * u.km},
    )

    result = CameraEfficiencyCalculator(telescope, site).calculate()
    assert len(result) == 801
    assert result.colnames[:3] == ["wl", "eff", "eff_atm"]
    assert result["eff"][200] == pytest.approx(0.09)
