#!/usr/bin/python3

import logging

from simtools.db import db_from_repo_handler

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_get_list_of_model_parameters(db_config):
    _pars_telescope_model = db_from_repo_handler.get_list_of_model_parameters(
        model_type="telescope_model",
        db_simulation_model_url=db_config["db_simulation_model_url"],
    )

    assert len(_pars_telescope_model)
    assert "telescope_axis_height" in _pars_telescope_model

    _pars_site_model = db_from_repo_handler.get_list_of_model_parameters(
        model_type="site_model",
        db_simulation_model_url=db_config["db_simulation_model_url"],
    )

    assert len(_pars_site_model)
    assert "epsg_code" in _pars_site_model
