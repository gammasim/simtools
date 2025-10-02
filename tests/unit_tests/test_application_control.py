"""Unit tests for application_control module."""

import logging
from unittest.mock import MagicMock

from simtools.application_control import get_application_label, startup_application


def test_get_application_label():
    """Test get_application_label function."""
    # Test with typical file path
    file_path = "/path/to/my_application.py"
    result = get_application_label(file_path)
    assert result == "my_application"

    # Test with different extension
    file_path = "/another/path/test_script.py"
    result = get_application_label(file_path)
    assert result == "test_script"

    # Test with no directory
    file_path = "simple_app.py"
    result = get_application_label(file_path)
    assert result == "simple_app"


def test_startup_application_basic():
    """Test startup_application function with basic configuration."""
    # Mock parse function
    mock_args_dict = {"log_level": "info", "test": True}
    mock_db_config = {"host": "localhost"}
    mock_parse_function = MagicMock(return_value=(mock_args_dict, mock_db_config))

    # Call startup_application
    app_context = startup_application(mock_parse_function)

    # Verify parse function was called
    mock_parse_function.assert_called_once()

    # Verify returned values
    assert app_context.args == mock_args_dict
    assert app_context.db_config == mock_db_config
    assert isinstance(app_context.logger, logging.Logger)
    assert app_context.io_handler is not None

    # Verify logger level was set
    assert app_context.logger.level == logging.INFO


def test_startup_application_without_io_handler():
    """Test startup_application function without IOHandler."""
    # Mock parse function
    mock_args_dict = {"log_level": "debug"}
    mock_db_config = {}
    mock_parse_function = MagicMock(return_value=(mock_args_dict, mock_db_config))

    # Call startup_application without IOHandler
    app_context = startup_application(mock_parse_function, setup_io_handler=False)

    # Verify parse function was called
    mock_parse_function.assert_called_once()

    # Verify returned values
    assert app_context.args == mock_args_dict
    assert app_context.db_config == mock_db_config
    assert isinstance(app_context.logger, logging.Logger)
    assert app_context.io_handler is None

    # Verify logger level was set to debug
    assert app_context.logger.level == logging.DEBUG


def test_startup_application_with_custom_logger_name():
    """Test startup_application function with custom logger name."""
    # Mock parse function
    mock_args_dict = {"log_level": "warning"}
    mock_db_config = {}
    mock_parse_function = MagicMock(return_value=(mock_args_dict, mock_db_config))

    # Call startup_application with custom logger name
    app_context = startup_application(
        mock_parse_function, logger_name="test_logger", setup_io_handler=False
    )

    # Verify logger name
    assert app_context.logger.name == "test_logger"
    assert app_context.logger.level == logging.WARNING
