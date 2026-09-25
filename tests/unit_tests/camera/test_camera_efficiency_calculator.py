"""Tests for the in-process camera-efficiency calculator."""

from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.table import QTable, Table

from simtools.camera.camera_efficiency_calculator import (
    CameraEfficiencyCalculator,
    _atmospheric_transmission,
    _column_values,
    _emission_altitude,
    _interpolate,
    _nearest,
    _parameter_table,
    _same_table_source,
    _spectral_curve,
    _table_from_file,
    _weights,
)


def test_column_values_handles_quantity_table_columns():
    table = QTable({"wavelength": [400.0, 500.0] * u.nm})

    np.testing.assert_allclose(_column_values(table, "wavelength", u.nm), [400.0, 500.0])


def test_interpolate_clips_and_sorts_support_points():
    result = _interpolate([10.0, 0.0], [2.0, 1.0], [-1.0, 5.0, 20.0])
    np.testing.assert_allclose(result, [1.0, 1.5, 2.0])


def test_interpolate_can_clip_outside_support_points():
    result = _interpolate([0.0, 10.0], [1.0, 2.0], [-1.0, 5.0, 11.0], clip=True)
    np.testing.assert_allclose(result, [0.0, 1.5, 0.0])


def test_interpolation_helpers_handle_empty_support_points():
    np.testing.assert_array_equal(_interpolate([], [], [400.0]), [0.0])
    np.testing.assert_array_equal(_nearest([], [], [10.0]), [0.0])


def test_same_table_source_compares_loaded_table_metadata(tmp_test_directory):
    source = Path(tmp_test_directory) / "lightguide.dat"
    first = Table(meta={"File": source})
    second = Table(meta={"File": source})
    different = Table(meta={"File": Path(tmp_test_directory) / "other.dat"})

    assert _same_table_source(first, second)
    assert not _same_table_source(first, different)
    assert not _same_table_source(first, Table())


def test_parameter_table_uses_validated_model_table():
    class Model:
        table = Table()

        def get_parameter_table(self, _):
            return self.table

    assert _parameter_table(Model(), "quantum_efficiency") is Model.table


def test_parameter_table_reads_nsb_correction_from_model_table():
    class Model:
        table = Table()

        def get_parameter_table(self, _):
            return self.table

    result = _parameter_table(Model(), "correct_nsb_spectrum_to_telescope_altitude")

    assert result is Model.table


def test_table_from_file_accepts_tables_and_reads_paths(tmp_test_directory):
    table = Table()
    assert _table_from_file(table) is table

    source = Path(tmp_test_directory) / "nsb.ecsv"
    expected = Table({"wavelength": [400.0], "flux": [1.0]})
    expected.write(source, format="ascii.ecsv")
    assert len(_table_from_file(source)) == 1


def test_weights_rejects_invalid_distribution_table():
    class Model:
        @staticmethod
        def get_parameter_table(_):
            return Table({"incidence_angle": [0.0]})

    with pytest.raises(ValueError, match="Invalid incidence-angle"):
        _weights(Model(), "incidence")


def test_spectral_curve_averages_angle_dependent_table():
    table = Table(
        {
            "wavelength": [400.0, 400.0, 500.0, 500.0] * u.nm,
            "incidence_angle": [0.0, 10.0, 0.0, 10.0] * u.deg,
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

    unweighted = _spectral_curve(
        table,
        np.array([400.0, 500.0]),
        candidates=("transmission", "efficiency"),
    )
    np.testing.assert_allclose(unweighted, [0.6, 0.4])


def test_spectral_curve_weights_each_angle_group_in_its_original_order():
    table = Table(
        {
            "wavelength": [400.0, 400.0, 500.0, 500.0] * u.nm,
            "incidence_angle": [0.0, 10.0, 10.0, 0.0] * u.deg,
            "efficiency": [0.8, 0.4, 0.2, 0.6],
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


def test_spectral_curve_rejects_tables_without_values():
    table = Table({"wavelength": [400.0] * u.nm})

    with pytest.raises(ValueError, match="efficiency"):
        _spectral_curve(table, np.array([400.0]))


def test_spectral_curve_averages_rpol_columns():
    table = Table(
        {
            "wavelength": [400.0, 500.0] * u.nm,
            "transmission_0deg": [0.8, 0.6],
            "transmission_10deg": [0.4, 0.2],
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
        candidates=("transmission", "efficiency"),
    )
    np.testing.assert_allclose(result, [0.5, 0.3])


def test_spectral_curve_uses_nearest_incidence_weight():
    table = Table(
        {
            "wavelength": [400.0, 400.0] * u.nm,
            "incidence_angle": [0.0, 10.0] * u.deg,
            "efficiency": [0.8, 0.4],
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
        np.array([400.0]),
        model=Model(),
        weighting_parameter="incidence",
    )
    assert result[0] == pytest.approx(0.5)


def test_camera_filter_uses_photon_incident_angle_distribution():
    class Model:
        def __init__(self):
            self.tables = {
                "camera_filter": Table(
                    {
                        "wavelength": [400.0, 400.0] * u.nm,
                        "incidence_angle": [0.0, 10.0] * u.deg,
                        "transmission": [0.8, 0.4],
                    }
                ),
                "camera_filter_photon_incident_angle": Table(
                    {
                        "incidence_angle": [0.0, 10.0] * u.deg,
                        "fraction": [0.25, 0.75],
                    }
                ),
            }

        def get_parameter_table(self, name):
            return self.tables[name]

    model = Model()
    calculator = CameraEfficiencyCalculator(model, model)

    result = calculator._camera_filter(np.array([400.0]))

    assert result[0] == pytest.approx(0.5)


def test_spectral_curve_rejects_an_incomplete_angle_grid():
    table = Table(
        {
            "wavelength": [400.0, 400.0, 500.0] * u.nm,
            "incidence_angle": [0.0, 10.0, 0.0] * u.deg,
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
    result = _atmospheric_transmission(table, np.array([350.0, 400.0]), 10.0**0.5, 2.0)
    np.testing.assert_allclose(result, [np.exp(-2.0), np.exp(-2.0)], rtol=1e-12)
    np.testing.assert_allclose(_atmospheric_transmission(table, [400.0], 0.5, 1.0), [0.0])


def test_emission_altitude_scales_xmax_by_airmass():
    profile = Table(
        {
            "altitude": [0.0, 10.0] * u.km,
            "thickness": [1000.0, 100.0] * (u.g / u.cm**2),
        }
    )
    assert _emission_altitude(profile, 500.0, 2.0) == pytest.approx(6.0206, rel=1e-4)
    assert _emission_altitude(profile, 2000.0, 1.0) == pytest.approx(0.0)


def test_emission_altitude_rejects_profile_without_positive_depth():
    profile = Table(
        {
            "altitude": [0.0, 10.0] * u.km,
            "thickness": [0.0, 0.0] * (u.g / u.cm**2),
        }
    )

    with pytest.raises(ValueError, match="positive thickness"):
        _emission_altitude(profile, 300.0, 1.0)


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
            self.design_model = "SST"

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
            "incidence_angle": [0.0, 10.0, 0.0, 10.0] * u.deg,
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
            {"incidence_angle": [0.0, 20.0] * u.deg, "efficiency": [1.0, 1.0]}
        ),
        "lightguide_efficiency_vs_wavelength": spectrum,
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


def test_funnel_does_not_apply_same_source_for_angle_and_wavelength():
    class Model:
        def __init__(self, tables):
            self.tables = tables
            self.design_model = "SST"

        def get_parameter_table(self, name):
            return self.tables[name]

        def get_parameter_value(self, name):
            return {"mirror_class": 1, "parabolic_dish": False}[name]

        def get_parameter_value_with_unit(self, name):
            assert name == "dish_shape_length"
            return 10.0 * u.m

        def get_telescope_effective_focal_length(self, *_args):
            return 10.0

    source = Path("lightguide.dat")
    angle = Table(
        {"incidence_angle": [0.0, 20.0] * u.deg, "efficiency": [0.8, 0.8]},
        meta={"File": source},
    )
    wavelength = Table(
        {"wavelength": [200.0, 1000.0] * u.nm, "efficiency": [0.5, 0.5]},
        meta={"File": source},
    )
    mirror = Table(
        {
            "mirror_x": [1.0] * u.cm,
            "mirror_y": [0.0] * u.cm,
            "mirror_diameter": [1.0] * u.cm,
            "shape_type": [0.0],
            "mirror_z": [0.0] * u.cm,
        }
    )
    tables = {
        "lightguide_efficiency_vs_incidence_angle": angle,
        "lightguide_efficiency_vs_wavelength": wavelength,
        "mirror_list": mirror,
    }
    calculator = CameraEfficiencyCalculator(Model(tables), Model(tables))

    _, funnel, _ = calculator._funnel_efficiency(np.array([400.0]), 1)

    assert funnel[0] == pytest.approx(0.8)


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
            self.design_model = "SST"

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
            {"incidence_angle": [0.0, 20.0] * u.deg, "efficiency": [0.8, 0.8]}
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
