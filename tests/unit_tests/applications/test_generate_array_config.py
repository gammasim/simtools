from unittest.mock import Mock

from simtools.applications import generate_array_config


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
