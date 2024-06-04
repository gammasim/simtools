#!/usr/bin/python3

import logging

import pytest

from simtools.model.site_model import SiteModel

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


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


def test_get_array_elements_for_layout(db_config, model_version, caplog):
    _north = SiteModel(
        site="North",
        mongo_db_config=db_config,
        label="testing-sitemodel",
        model_version=model_version,
    )

    assert isinstance(_north.get_array_elements_for_layout("testlayout"), list)
    assert len(_north.get_array_elements_for_layout("testlayout")) == 13
    assert "LSTN-01" in _north.get_array_elements_for_layout("testlayout")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError):
            _north.get_array_elements_for_layout("not_a_layout")
        assert "Array layout 'not_a_layout' not found in site model." in caplog.text
