"""Unit tests for application_control module."""

import logging
from unittest.mock import MagicMock, patch

from simtools.application_control import (
    _resolve_model_version_to_latest_patch,
    get_application_label,
    startup_application,
)


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


def test_resolve_model_version_to_latest_patch_no_model_version():
    """Test _resolve_model_version_to_latest_patch when model_version is not in args."""

    args_dict = {"log_level": "info"}
    db_config = {"host": "localhost"}
    logger = logging.getLogger("test")

    _resolve_model_version_to_latest_patch(args_dict, db_config, logger)

    assert "model_version" not in args_dict


def test_resolve_model_version_to_latest_patch_no_db_config():
    """Test _resolve_model_version_to_latest_patch when db_config is None."""

    args_dict = {"model_version": "6.0"}
    db_config = None
    logger = logging.getLogger("test")

    _resolve_model_version_to_latest_patch(args_dict, db_config, logger)

    assert args_dict["model_version"] == "6.0"


def test_resolve_model_version_to_latest_patch_full_version():
    """Test _resolve_model_version_to_latest_patch when model_version is already full."""

    args_dict = {"model_version": "6.0.1"}
    db_config = {"host": "localhost"}
    logger = logging.getLogger("test")

    with patch("simtools.application_control.db_handler.DatabaseHandler") as mock_db_class:
        with patch("simtools.application_control.version.version_kind") as mock_version_kind:
            with patch(
                "simtools.application_control.version.MAJOR_MINOR_PATCH", "major.minor.patch"
            ):
                mock_version_kind.return_value = "major.minor.patch"
                _resolve_model_version_to_latest_patch(args_dict, db_config, logger)

    assert args_dict["model_version"] == "6.0.1"
    mock_db_class.assert_not_called()


def test_resolve_model_version_to_latest_patch_resolves_to_latest():
    """Test _resolve_model_version_to_latest_patch resolves to latest patch version."""

    args_dict = {"model_version": "6.0"}
    db_config = {"host": "localhost"}
    logger = logging.getLogger("test")

    mock_db = MagicMock()
    mock_db.get_model_versions.return_value = ["6.0.0", "6.0.1", "6.0.2"]

    with patch("simtools.application_control.db_handler.DatabaseHandler", return_value=mock_db):
        with patch("simtools.application_control.version.version_kind", return_value="MAJOR_MINOR"):
            with patch(
                "simtools.application_control.version.resolve_version_to_latest_patch",
                return_value="6.0.2",
            ) as mock_resolve:
                _resolve_model_version_to_latest_patch(args_dict, db_config, logger)

                mock_resolve.assert_called_once_with("6.0", ["6.0.0", "6.0.1", "6.0.2"])
                assert args_dict["model_version"] == "6.0.2"
