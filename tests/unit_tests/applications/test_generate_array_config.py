from unittest.mock import Mock

import astropy.units as u
import pytest

from simtools.applications import generate_array_config
from simtools.simulation.nsb import NsbSettings


def test_main_passes_registered_array_element_list(mocker):
    context = Mock()
    context.args = {
        "label": "selected-array",
        "model_version": "7.0.0",
        "site": "North",
        "array_layout_name": None,
        "array_element_list": ["LSTN-01", "MSTN-01"],
    }
    context.model_reader = Mock()
    application = mocker.Mock()
    application.start.return_value = context
    mocker.patch.object(generate_array_config, "APPLICATION", application)
    mock_array_model = mocker.patch.object(generate_array_config, "ArrayModel")
    mock_get_writer = mocker.patch.object(generate_array_config, "get_model_writer")

    generate_array_config.main()

    mock_array_model.assert_called_once_with(
        label="selected-array",
        model_version="7.0.0",
        site="North",
        layout_name=None,
        array_elements=["LSTN-01", "MSTN-01"],
        model_reader=context.model_reader,
        context=context,
    )
    mock_array_model.return_value.print_telescope_list.assert_called_once()
    mock_get_writer.assert_called_once_with(mock_array_model.return_value, context=context)
    mock_get_writer.return_value.export_all_simtel_config_files.assert_called_once()


@pytest.mark.parametrize(
    ("zenith_angle", "expected_settings"),
    [(None, None), (40.0 * u.deg, NsbSettings(40.0, None, True))],
)
def test_main_sets_nsb_settings_for_zenith_angle(mocker, zenith_angle, expected_settings):
    context = Mock()
    context.args = {"label": "test", "model_version": "7.0.0", "zenith_angle": zenith_angle}
    application = mocker.Mock()
    application.start.return_value = context
    mocker.patch.object(generate_array_config, "APPLICATION", application)
    mocker.patch.object(generate_array_config, "ArrayModel")
    writer = mocker.patch.object(generate_array_config, "get_model_writer").return_value
    writer.nsb_settings = None

    generate_array_config.main()

    assert writer.nsb_settings == expected_settings
    writer.export_all_simtel_config_files.assert_called_once()
