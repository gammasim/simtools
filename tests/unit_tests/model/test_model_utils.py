#!/usr/bin/python3

import pytest

from simtools.model import model_utils


@pytest.mark.parametrize(
    ("telescope_name", "expected"),
    [
        ("LSTN-01", False),
        ("MSTN-01", False),
        ("MSTS-01", False),
        ("SSTS-01", True),
        ("SCTS-25", True),
    ],
)
def test_is_two_mirror_telescope(telescope_name, expected):
    assert model_utils.is_two_mirror_telescope(telescope_name) == expected


@pytest.mark.parametrize(
    ("pars", "off_axis", "expected"),
    [
        ([0.8, 0, 0.0, 0.0, 0.0], 0.0, 0.8),
        ([0.898, 1, 0.016, 4.136, 1.705, 0.0], 0.0, 0.898),
        ([0.898, 1, 0.016, 4.136, 1.705, 0.0], 2.0, 0.8938578),
    ],
)
def test_compute_telescope_transmission(pars, off_axis, expected):
    assert model_utils.compute_telescope_transmission(pars, off_axis) == pytest.approx(expected)


@pytest.mark.parametrize(("site", "telescope_name"), [("North", "LSTN-01"), ("South", "MSTN-01")])
def test_initialize_simulation_models(mocker, site, telescope_name):
    mock_tel_model = mocker.patch("simtools.model.model_utils.TelescopeModel")
    mocker.patch("simtools.model.model_utils.SiteModel")
    mocker.patch("simtools.model.model_utils.CalibrationModel")
    mock_tel_model.return_value.get_calibration_device_name.return_value = None

    telescope_model, site_model, calibration_model = model_utils.initialize_simulation_models(
        label="test_label",
        site=site,
        telescope_name=telescope_name,
        model_version="test_version",
    )
    assert telescope_model == mock_tel_model.return_value
    assert site_model is not None
    assert calibration_model is None


def test_initialize_simulation_models_with_calibration_device(mocker):
    mocker.patch("simtools.model.model_utils.TelescopeModel")
    mocker.patch("simtools.model.model_utils.SiteModel")
    mock_cal_model = mocker.patch("simtools.model.model_utils.CalibrationModel")

    _, _, calibration_model = model_utils.initialize_simulation_models(
        label="test_label",
        site="North",
        telescope_name="LSTN-01",
        model_version="test_version",
        calibration_device_name="flasher_device",
    )

    mock_cal_model.assert_called_once()
    assert calibration_model == mock_cal_model.return_value


def test_read_overwrite_model_parameter_dict_with_file(mocker):
    mocker.patch("simtools.model.model_utils.ascii_handler.collect_data_from_file")
    mock_schema = mocker.patch("simtools.model.model_utils.schema.validate_dict_using_schema")
    mock_schema.return_value = {"changes": {"param1": "value1"}}

    result = model_utils.read_overwrite_model_parameter_dict("test_file.yml")

    assert result == {"param1": "value1"}


@pytest.mark.parametrize(
    ("layout_name", "site", "model_version"),
    [
        ("test_layout", "North", "1.0.0"),
        (["test_layout"], "South", "2.0.0"),
    ],
)
def test_get_array_elements_for_layout(mocker, layout_name, site, model_version):
    mock_site_model = mocker.patch("simtools.model.model_utils.SiteModel")
    mock_site_model.return_value.get_array_elements_for_layout.return_value = [
        "element1",
        "element2",
    ]

    result = model_utils.get_array_elements_for_layout(
        layout_name, site=site, model_version=model_version
    )

    assert result == ["element1", "element2"]
    mock_site_model.assert_called_once()


@pytest.mark.parametrize(
    "layout_name",
    [
        "",
        ["layout1", "layout2"],
    ],
)
def test_get_array_elements_for_layout_invalid(mocker, layout_name):
    mocker.patch("simtools.model.model_utils.SiteModel")

    with pytest.raises(ValueError, match="Single array layout name must be provided"):
        model_utils.get_array_elements_for_layout(layout_name, site="North", model_version="1.0.0")
