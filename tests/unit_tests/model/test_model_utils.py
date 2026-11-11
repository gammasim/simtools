#!/usr/bin/python3

import json

import pytest

from simtools.data_model.model_data_writer import ModelDataWriter
from simtools.model import model_utils


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


def test_read_overwrite_model_parameter_dict_with_dict_no_changes_key():
    result = model_utils.read_overwrite_model_parameter_dict({"param1": "value1"})

    assert result == {"param1": "value1"}


def test_read_overwrite_model_parameter_dict_with_json_string():
    overwrite_str = '{"changes": {"param1": "value1"}}'

    result = model_utils.read_overwrite_model_parameter_dict(overwrite_str)

    assert result == {"param1": "value1"}


@pytest.mark.parametrize("value", [12, [0.0075, 0.22, 0.022], {"value": [1, 2]}])
def test_read_overwrite_model_parameter_document(tmp_test_directory, value):
    output_file = tmp_test_directory / "parameter.json"
    ModelDataWriter.write_model_parameter_json(
        {
            "instrument": "LSTN-01",
            "parameter": "mirror_reflection_random_angle",
            "value": value,
            "unit": ["deg", "dimensionless", "deg"],
            "parameter_version": "3.0.0",
            "schema_version": "0.4.0",
            "file": False,
        },
        output_file,
    )

    result = model_utils.read_overwrite_model_parameter_dict(str(output_file))

    assert result == {"LSTN-01": {"mirror_reflection_random_angle": {"value": value}}}


@pytest.mark.parametrize("wrapped", [False, True])
def test_read_overwrite_model_parameter_mapping_file(tmp_test_directory, wrapped):
    overrides = {"LSTN-01": {"mirror_reflection_random_angle": [0.0075, 0.22, 0.022]}}
    data = {"changes": overrides} if wrapped else overrides
    output_file = tmp_test_directory / "overrides.json"
    output_file.write_text(json.dumps(data), encoding="utf-8")

    assert model_utils.read_overwrite_model_parameter_dict(str(output_file)) == overrides


def test_model_parameter_document_applied_to_model(tmp_test_directory, model_version, run_context):
    output_file = tmp_test_directory / "parameter.json"
    ModelDataWriter.write_model_parameter_json(
        {
            "instrument": "LSTN-01",
            "parameter": "num_gains",
            "value": 1,
            "parameter_version": "99.0.0",
        },
        output_file,
    )
    overrides = model_utils.read_overwrite_model_parameter_dict(str(output_file))

    model = model_utils.TelescopeModel(
        site="North",
        telescope_name="LSTN-01",
        model_version=model_version,
        overwrite_model_parameter_dict=overrides,
        context=run_context,
    )

    assert model.parameters["num_gains"]["value"] == 1
    assert model.parameters["num_gains"]["parameter_version"] != "99.0.0"


def test_read_overwrite_model_parameter_dict_with_file_no_changes_key(mocker):
    mock_collect = mocker.patch("simtools.model.model_utils.ascii_handler.collect_data_from_file")
    mock_collect.return_value = {"param1": "value1"}

    result = model_utils.read_overwrite_model_parameter_dict("test_file.yml")

    assert result == {"param1": "value1"}


def test_read_overwrite_model_parameter_dict_with_file_non_dict_data(mocker):
    mocker.patch(
        "simtools.model.model_utils.ascii_handler.collect_data_from_file",
        return_value=["not", "a", "dict"],
    )

    result = model_utils.read_overwrite_model_parameter_dict("test_file.yml")

    assert result == {}


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
