#!/usr/bin/python3

import logging

import pytest

from simtools.db import db_from_repo_handler

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_update_parameters_from_repo(caplog, db_config):
    with caplog.at_level(logging.DEBUG):
        assert (
            db_from_repo_handler.update_model_parameters_from_repo(
                parameters={},
                site="North",
                telescope_name="MSTN-01",
                model_version="2024-02-01",
                parameter_collection="telescopes",
                db_simulation_model_url=None,
                db_simulation_model="verified_model",
            )
            == {}
        )
        assert "No repository specified, skipping" in caplog.text

    if (
        db_config["db_simulation_model_url"] is None
        or len(db_config["db_simulation_model_url"]) == 0
    ):
        db_config["db_simulation_model_url"] = (
            "https://gitlab.cta-observatory.org/cta-science/simulations/"
            "simulation-model/model_parameters/-/raw/main/"
        )

    _pars_telescope_model = ["telescope_axis_height", "telescope_sphere_radius"]

    for _tel in ["MSTN-01", "MSTN-design"]:
        _pars_mstn = db_from_repo_handler.update_model_parameters_from_repo(
            parameters=dict.fromkeys(_pars_telescope_model, None),
            site="North",
            telescope_name=_tel,
            model_version="2024-02-01",
            parameter_collection="telescopes",
            db_simulation_model_url=db_config["db_simulation_model_url"],
            db_simulation_model="verified_model",
        )
        assert len(_pars_mstn) > 0
        assert "telescope_axis_height" in _pars_mstn
        for key in ["value", "unit", "site"]:
            assert key in _pars_mstn["telescope_axis_height"]

    _pars_site_model = [
        "corsika_observation_level",
        "geomag_horizontal",
        "reference_point_utm_east",
    ]

    _pars_south = db_from_repo_handler.update_model_parameters_from_repo(
        parameters=dict.fromkeys(_pars_site_model, None),
        site="South",
        telescope_name=None,
        model_version="2024-02-01",
        parameter_collection="site",
        db_simulation_model_url=db_config["db_simulation_model_url"],
        db_simulation_model="verified_model",
    )
    assert len(_pars_south) > 0

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError):
            db_from_repo_handler.update_model_parameters_from_repo(
                parameters=dict.fromkeys(_pars_telescope_model, None),
                site="North",
                telescope_name="MSTN-01",
                model_version="2024-02-01",
                parameter_collection="not_a_collection",
                db_simulation_model_url=db_config["db_simulation_model_url"],
                db_simulation_model="verified_model",
            )
        assert "Unknown parameter collection" in caplog.text

    # Test with a parameter that is not in the repository (no error should be raised)
    _pars_site_model.append("not_a_parameter")
    db_from_repo_handler.update_model_parameters_from_repo(
        parameters=dict.fromkeys(_pars_site_model, None),
        site="South",
        telescope_name=None,
        model_version="2024-02-01",
        parameter_collection="site",
        db_simulation_model_url=db_config["db_simulation_model_url"],
        db_simulation_model="verified_model",
    )


def test_update_telescope_parameters_from_repo(db_config):
    _pars_telescope_model = ["telescope_axis_height", "telescope_sphere_radius"]
    _pars = db_from_repo_handler.update_model_parameters_from_repo(
        parameters=dict.fromkeys(_pars_telescope_model, None),
        site="North",
        telescope_name="MSTN-01",
        parameter_collection="telescopes",
        model_version="2024-02-01",
        db_simulation_model_url=db_config["db_simulation_model_url"],
    )
    assert len(_pars) > 0


def test_update_site_parameters_from_repo(db_config):
    _pars_site_model = [
        "corsika_observation_level",
        "geomag_horizontal",
        "reference_point_utm_east",
    ]
    _pars = db_from_repo_handler.update_model_parameters_from_repo(
        parameters=dict.fromkeys(_pars_site_model, None),
        site="South",
        telescope_name=None,
        parameter_collection="site",
        model_version="2024-02-01",
        db_simulation_model_url=db_config["db_simulation_model_url"],
    )
    assert len(_pars) > 0
