#!/usr/bin/python3

import logging

import astropy.units as u
import pytest
from astropy.table import QTable

from simtools.model.site_model import SiteModel

logger = logging.getLogger()


def test_get_corsika_site_parameters(model_version, run_context):
    _north = SiteModel(
        context=run_context,
        site="North",
        label="testing-sitemodel",
        model_version=model_version,
    )

    assert "corsika_observation_level" in _north.get_corsika_site_parameters()

    assert "ARRANG" in _north.get_corsika_site_parameters(config_file_style=True)


def test_get_corsika_site_parameters_with_model_directory(array_model_north, run_context):
    model_directory = array_model_north.get_config_directory()
    corsika_site_parameters = array_model_north.site_model.get_corsika_site_parameters(
        config_file_style=True, model_directory=model_directory
    )
    assert "model/" in str(corsika_site_parameters["IACT ATMOFILE"][0])


def test_get_array_elements_for_layout_single_telescope(model_version, run_context):
    _north = SiteModel(
        context=run_context, site="North", label="testing-sitemodel", model_version=model_version
    )
    _south = SiteModel(
        context=run_context, site="South", label="testing-sitemodel", model_version=model_version
    )

    # Valid North telescope used as single-telescope layout
    result = _north.get_array_elements_for_layout("MSTN-05")
    assert result == ["MSTN-05"]

    result = _north.get_array_elements_for_layout("LSTN-01")
    assert result == ["LSTN-01"]

    # Valid South telescope used as single-telescope layout
    result = _south.get_array_elements_for_layout("MSTS-05")
    assert result == ["MSTS-05"]

    # Wrong-site telescope raises ValueError
    with pytest.raises(
        ValueError, match=r"Array layout 'MSTS-05' not found in 'North' site model."
    ):
        _north.get_array_elements_for_layout("MSTS-05")

    # Design-type telescope raises ValueError
    with pytest.raises(
        ValueError, match=r"Array layout 'LSTN-design' not found in 'North' site model."
    ):
        _north.get_array_elements_for_layout("LSTN-design")

    # Calibration device raises ValueError
    with pytest.raises(
        ValueError, match=r"Array layout 'ILLN-01' not found in 'North' site model."
    ):
        _north.get_array_elements_for_layout("ILLN-01")


def test_get_list_of_array_layouts(model_version, run_context):
    _north = SiteModel(
        context=run_context,
        site="North",
        label="testing-sitemodel",
        model_version=model_version,
    )

    assert isinstance(_north.get_list_of_array_layouts(), list)
    assert "test_layout" in _north.get_list_of_array_layouts()


def test_get_nsb_integrated_flux_returns_zero_for_empty_wavelength_range(
    model_version, mocker, run_context
):
    site_model = SiteModel(
        context=run_context,
        site="South",
        label="testing-sitemodel",
        model_version=model_version,
    )

    table = QTable()
    table["wavelength"] = [700, 800] * u.nm
    table["differential_photon_rate"] = [1, 2] * (1 / (u.nm * u.cm**2 * u.ns * u.sr))
    get_parameter_table = mocker.patch.object(site_model, "get_parameter_table", return_value=table)

    result = site_model.get_nsb_integrated_flux()

    assert result == pytest.approx(0.0)
    get_parameter_table.assert_called_once_with("nsb_spectrum")
