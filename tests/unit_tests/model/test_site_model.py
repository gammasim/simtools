#!/usr/bin/python3

import logging

import pytest

from simtools.model.site_model import SiteModel

logger = logging.getLogger()


def test_site_model(db_config, model_version):
    _south = SiteModel(
        site="South",
        mongo_db_config=db_config,
        label="testing-sitemodel",
        model_version=model_version,
    )

    assert isinstance(_south.get_reference_point(), dict)
    for key in ["center_altitude", "center_northing", "center_easting", "epsg_code"]:
        assert key in _south.get_reference_point()

    _pars = _south.get_simtel_parameters(_south._parameters)
    assert "altitude" in _pars
    assert isinstance(_pars["altitude"], float)


def test_get_corsika_site_parameters(db_config, model_version):
    _north = SiteModel(
        site="North",
        mongo_db_config=db_config,
        label="testing-sitemodel",
        model_version=model_version,
    )

    assert "corsika_observation_level" in _north.get_corsika_site_parameters()

    assert "ARRANG" in _north.get_corsika_site_parameters(config_file_style=True)


def test_get_corsika_site_parameters_with_model_directory(array_model):
    """Test that the amtospheric profile file is provided with the model directory."""
    model_directory = array_model.get_config_directory()
    corsika_site_parameters = array_model.site_model.get_corsika_site_parameters(
        config_file_style=True, model_directory=model_directory
    )
    assert "test/model/" in str(corsika_site_parameters["IACT ATMOFILE"][0])


def test_get_array_elements_for_layout(db_config, model_version):
    _north = SiteModel(
        site="North",
        mongo_db_config=db_config,
        label="testing-sitemodel",
        model_version=model_version,
    )

    assert isinstance(_north.get_array_elements_for_layout("test_layout"), list)
    assert len(_north.get_array_elements_for_layout("test_layout")) == 13
    assert "LSTN-01" in _north.get_array_elements_for_layout("test_layout")

    with pytest.raises(
        ValueError, match="Array layout 'not_a_layout' not found in 'North' site model."
    ):
        _north.get_array_elements_for_layout("not_a_layout")


def test_get_list_of_array_layouts(db_config, model_version):
    _north = SiteModel(
        site="North",
        mongo_db_config=db_config,
        label="testing-sitemodel",
        model_version=model_version,
    )

    assert isinstance(_north.get_list_of_array_layouts(), list)
    assert "test_layout" in _north.get_list_of_array_layouts()


def test_export_atmospheric_transmission_file(db_config, model_version, tmp_path, mocker):
    _south = SiteModel(
        site="South",
        mongo_db_config=db_config,
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
