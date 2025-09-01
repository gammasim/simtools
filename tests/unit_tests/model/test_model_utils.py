#!/usr/bin/python3

import pytest

from simtools.model import model_utils


def test_is_two_mirror_telescope():
    assert not model_utils.is_two_mirror_telescope("LSTN-01")
    assert not model_utils.is_two_mirror_telescope("MSTN-01")
    assert not model_utils.is_two_mirror_telescope("MSTS-01")
    assert model_utils.is_two_mirror_telescope("SSTS-01")
    assert model_utils.is_two_mirror_telescope("SCTS-25")


def test_compute_telescope_transmission():
    pars = [0.8, 0, 0.0, 0.0, 0.0]
    off_axis = 0.0
    assert model_utils.compute_telescope_transmission(pars, off_axis) == pytest.approx(pars[0])

    pars = [0.898, 1, 0.016, 4.136, 1.705, 0.0]
    off_axis = 0.0
    assert model_utils.compute_telescope_transmission(pars, off_axis) == pytest.approx(pars[0])

    pars = [0.898, 1, 0.016, 4.136, 1.705, 0.0]
    off_axis = 2.0
    assert model_utils.compute_telescope_transmission(pars, off_axis) == pytest.approx(0.8938578)


@pytest.fixture
def mock_db_config():
    return {"host": "localhost", "port": 27017}


@pytest.mark.parametrize(("site", "telescope_name"), [("North", "LSTN-01"), ("South", "MSTN-01")])
def test_initialize_simulation_models(mocker, mock_db_config, site, telescope_name):
    mock_tel_model = mocker.patch("simtools.model.model_utils.TelescopeModel")
    mock_site_model = mocker.patch("simtools.model.model_utils.SiteModel")

    label = "test_label"
    model_version = "test_version"

    model_utils.initialize_simulation_models(
        label=label,
        db_config=mock_db_config,
        site=site,
        telescope_name=telescope_name,
        model_version=model_version,
    )

    mock_tel_model.assert_called_once_with(
        site=site,
        telescope_name=telescope_name,
        mongo_db_config=mock_db_config,
        model_version=model_version,
        label=label,
    )

    mock_site_model.assert_called_once_with(
        site=site, model_version=model_version, mongo_db_config=mock_db_config, label=label
    )

    mock_tel_model.return_value.export_model_files.assert_called_once()
    mock_site_model.return_value.export_model_files.assert_called_once()
