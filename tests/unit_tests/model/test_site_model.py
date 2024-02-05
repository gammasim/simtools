#!/usr/bin/python3

import logging

from simtools.model.site_model import SiteModel

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_site_model(db_config):
    _south = SiteModel(site="South", mongo_db_config=db_config, label="testing-sitemodel")

    assert isinstance(_south.get_reference_point(), dict)
    for key in ["center_altitude", "center_northing", "center_easting", "epsg_code"]:
        assert key in _south.get_reference_point()

    assert "corsika_observation_level" in _south.get_corsika_site_parameters()

    _pars = _south.get_simtel_parameters()
    assert "altitude" in _pars
    assert isinstance(_pars["altitude"], float)
