#!/usr/bin/python3

import logging

import astropy.units as u
import pytest

from simtools.model.site_model import SiteModel

logger = logging.getLogger()


def test_site_model(model_version):
    _south = SiteModel(
        site="South",
        label="testing-sitemodel",
        model_version=model_version,
    )

    assert isinstance(_south.get_reference_point(), dict)
    for key in ["center_altitude", "center_northing", "center_easting", "epsg_code"]:
        assert key in _south.get_reference_point()

    assert "reference_point_altitude" in _south.parameters.keys()
    assert isinstance(_south.parameters["reference_point_altitude"]["value"], float)


def test_get_corsika_site_parameters(model_version):
    _north = SiteModel(
        site="North",
        label="testing-sitemodel",
        model_version=model_version,
    )

    assert "corsika_observation_level" in _north.get_corsika_site_parameters()

    assert "ARRANG" in _north.get_corsika_site_parameters(config_file_style=True)


def test_get_corsika_site_parameters_with_model_directory(array_model_north):
    """Test that the atmospheric profile file is provided with the model directory."""
    model_directory = array_model_north.get_config_directory()
    corsika_site_parameters = array_model_north.site_model.get_corsika_site_parameters(
        config_file_style=True, model_directory=model_directory
    )
    assert "model/" in str(corsika_site_parameters["IACT ATMOFILE"][0])


def test_get_array_elements_for_layout(model_version):
    _north = SiteModel(
        site="North",
        label="testing-sitemodel",
        model_version=model_version,
    )

    assert isinstance(_north.get_array_elements_for_layout("test_layout"), list)
    assert len(_north.get_array_elements_for_layout("test_layout")) == 13
    assert "LSTN-01" in _north.get_array_elements_for_layout("test_layout")

    with pytest.raises(
        ValueError, match=r"Array layout 'not_a_layout' not found in 'North' site model."
    ):
        _north.get_array_elements_for_layout("not_a_layout")


def test_get_list_of_array_layouts(model_version):
    _north = SiteModel(
        site="North",
        label="testing-sitemodel",
        model_version=model_version,
    )

    assert isinstance(_north.get_list_of_array_layouts(), list)
    assert "test_layout" in _north.get_list_of_array_layouts()


def test_export_atmospheric_transmission_file(model_version, tmp_path, mocker):
    _south = SiteModel(
        site="South",
        label="testing-sitemodel",
        model_version=model_version,
    )

    mocker.patch.object(_south, "get_parameter_value", return_value="test_atmospheric_profile")
    mocker.patch.object(_south.db, "export_model_files")

    model_directory = tmp_path / "model"
    model_directory.mkdir()

    _south.export_atmospheric_transmission_file(model_directory)

    _south.db.export_model_files.assert_called_once_with(
        parameters={
            "atmospheric_transmission_file": {
                "value": "test_atmospheric_profile",
                "file": True,
            }
        },
        dest=model_directory,
    )


def test_get_nsb_integrated_flux(model_version, mocker):
    _south = SiteModel(
        site="South",
        label="testing-sitemodel",
        model_version=model_version,
    )

    # Build a minimal fake table object with distinct columns to avoid
    # MagicMock __getitem__ returning the same mock for different keys.
    class Col:
        def __init__(self, q):
            self.quantity = q

    class FakeTable(dict):
        def sort(self, key):
            # no-op for the simple test
            return None

    wl_q = [300, 400, 500, 600, 700] * u.nm
    rate_q = [1, 2, 3, 4, 5] * (1 / (u.nm * u.cm**2 * u.ns * u.sr))

    mock_table = FakeTable()
    mock_table["wavelength"] = Col(wl_q)
    mock_table["differential photon rate"] = Col(rate_q)

    mocker.patch.object(_south.db, "get_ecsv_file_as_astropy_table", return_value=mock_table)
    mocker.patch.object(_south, "get_parameter_value", return_value="test_nsb_spectrum.ecsv")
    result = _south.get_nsb_integrated_flux(wavelength_min=300 * u.nm, wavelength_max=650 * u.nm)

    assert isinstance(result, float)
    assert result > 0
