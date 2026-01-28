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


@pytest.mark.parametrize(("site", "telescope_name"), [("North", "LSTN-01"), ("South", "MSTN-01")])
def test_initialize_simulation_models(mocker, site, telescope_name):
    mock_tel_model = mocker.patch("simtools.model.model_utils.TelescopeModel")
    mock_site_model = mocker.patch("simtools.model.model_utils.SiteModel")

    label = "test_label"
    model_version = "test_version"

    model_utils.initialize_simulation_models(
        label=label,
        site=site,
        telescope_name=telescope_name,
        model_version=model_version,
    )

    mock_tel_model.assert_called_once_with(
        site=site,
        telescope_name=telescope_name,
        model_version=model_version,
        label=label,
        overwrite_model_parameter_dict={},
    )

    mock_site_model.assert_called_once_with(
        site=site,
        model_version=model_version,
        label=label,
        overwrite_model_parameter_dict={},
    )

    mock_tel_model.return_value.export_model_files.assert_called_once()
    mock_site_model.return_value.export_model_files.assert_called_once()


def test_initialize_simulation_models_with_calibration_device(mocker):
    """Test initialize_simulation_models when calibration_device_name is provided."""
    mocker.patch("simtools.model.model_utils.TelescopeModel")
    mocker.patch("simtools.model.model_utils.SiteModel")
    mock_cal_model = mocker.patch("simtools.model.model_utils.CalibrationModel")

    label = "test_label"
    model_version = "test_version"
    site = "North"
    telescope_name = "LSTN-01"
    calibration_device_name = "flasher_device"

    _, _, calibration_model = model_utils.initialize_simulation_models(
        label=label,
        site=site,
        telescope_name=telescope_name,
        model_version=model_version,
        calibration_device_name=calibration_device_name,
    )

    mock_cal_model.assert_called_once_with(
        site=site,
        calibration_device_model_name=calibration_device_name,
        model_version=model_version,
        label=label,
        overwrite_model_parameter_dict={},
    )
    assert calibration_model == mock_cal_model.return_value


def test_initialize_simulation_models_without_calibration_device(mocker):
    """Test initialize_simulation_models when calibration_device_name is None."""
    mocker.patch("simtools.model.model_utils.TelescopeModel")
    mocker.patch("simtools.model.model_utils.SiteModel")
    mock_cal_model = mocker.patch("simtools.model.model_utils.CalibrationModel")

    label = "test_label"
    model_version = "test_version"
    site = "North"
    telescope_name = "LSTN-01"

    _, _, calibration_model = model_utils.initialize_simulation_models(
        label=label,
        site=site,
        telescope_name=telescope_name,
        model_version=model_version,
        calibration_device_name=None,
    )

    mock_cal_model.assert_not_called()
    assert calibration_model is None
