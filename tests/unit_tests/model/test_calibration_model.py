#!/usr/bin/python3

import logging
from unittest.mock import Mock, patch

from simtools.model.calibration_model import CalibrationModel


@patch("simtools.db.db_handler.DatabaseHandler")
def test_calibration_model_init(mock_db_handler, caplog):
    """Test CalibrationModel initialization."""
    mock_db_instance = Mock()
    mock_db_handler.return_value = mock_db_instance
    mock_db_instance.get_design_model.return_value = {}
    mock_db_instance.get_model_parameters.return_value = {}
    mock_db_instance.get_simulation_configuration_parameters.return_value = {}

    with caplog.at_level(logging.DEBUG):
        model = CalibrationModel(
            site="North",
            calibration_device_model_name="ILLN-01",
            model_version="1.0.0",
        )

    assert model.site == "North"
    assert model.name == "ILLN-01"
    assert "Init CalibrationModel North ILLN-01" in caplog.text


@patch("simtools.db.db_handler.DatabaseHandler")
def test_calibration_model_init_with_label(mock_db_handler):
    """Test CalibrationModel initialization with label."""
    mock_db_instance = Mock()
    mock_db_handler.return_value = mock_db_instance
    mock_db_instance.get_design_model.return_value = {}
    mock_db_instance.get_model_parameters.return_value = {}
    mock_db_instance.get_simulation_configuration_parameters.return_value = {}

    model = CalibrationModel(
        site="South",
        calibration_device_model_name="ILLS-01",
        model_version="1.0.0",
        label="test_label",
    )

    assert model.label == "test_label"
