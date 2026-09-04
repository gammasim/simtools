"""Tests for the validate_file_using_schema application."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

from simtools.applications import validate_file_using_schema


def test_application_does_not_initialize_model_reader():
    """Schema validation does not need simulation-model data."""
    assert validate_file_using_schema.APPLICATION.initialize_model_reader is False


@patch("simtools.applications.validate_file_using_schema.output_validator")
@patch("simtools.application.definition.ApplicationDefinition.start")
def test_main_validates_reduced_event_data(mock_start, mock_output_validator):
    logger = Mock()
    mock_start.return_value = SimpleNamespace(
        args={
            "file_name": "reduced_event_data.hdf5",
            "file_directory": None,
            "schema_file": None,
            "data_type": "reduced_event_data",
            "check_exact_data_type": False,
        },
        logger=logger,
    )

    validate_file_using_schema.main()

    mock_output_validator.validate_reduced_event_data_file.assert_called_once_with(
        "reduced_event_data.hdf5"
    )
    logger.info.assert_called_once_with(
        "Successful validation of reduced event data reduced_event_data.hdf5"
    )


@patch("simtools.applications.validate_file_using_schema.output_validator")
@patch("simtools.application.definition.ApplicationDefinition.start")
def test_main_validates_trigger_histograms(mock_start, mock_output_validator):
    logger = Mock()
    mock_start.return_value = SimpleNamespace(
        args={
            "file_name": "trigger_histograms.hdf5",
            "file_directory": None,
            "schema_file": None,
            "data_type": "trigger_histograms",
            "check_exact_data_type": False,
        },
        logger=logger,
    )

    validate_file_using_schema.main()

    mock_output_validator.validate_trigger_histogram_file.assert_called_once_with(
        "trigger_histograms.hdf5"
    )
    logger.info.assert_called_once_with(
        "Successful validation of trigger histograms trigger_histograms.hdf5"
    )
