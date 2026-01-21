"""Unit tests for application_control module."""

import logging
from unittest.mock import MagicMock, patch

from simtools.application_control import (
    _resolve_model_version_to_latest_patch,
    _version_info,
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
    logger = logging.getLogger("test")

    _resolve_model_version_to_latest_patch(args_dict, logger)

    assert "model_version" not in args_dict


def test_resolve_model_version_to_latest_patch_full_version():
    """Test _resolve_model_version_to_latest_patch when model_version is already full."""

    args_dict = {"model_version": "6.0.1"}
    logger = logging.getLogger("test")

    with patch("simtools.application_control.db_handler.DatabaseHandler") as mock_db_class:
        with patch("simtools.application_control.version.version_kind") as mock_version_kind:
            with patch(
                "simtools.application_control.version.MAJOR_MINOR_PATCH", "major.minor.patch"
            ):
                mock_version_kind.return_value = "major.minor.patch"
                _resolve_model_version_to_latest_patch(args_dict, logger)

    assert args_dict["model_version"] == "6.0.1"
    mock_db_class.assert_not_called()


def test_resolve_model_version_to_latest_patch_resolves_to_latest():
    """Test _resolve_model_version_to_latest_patch resolves to latest patch version."""

    args_dict = {"model_version": "6.0"}
    logger = logging.getLogger("test")

    mock_db = MagicMock()
    mock_db.get_model_versions.return_value = ["6.0.0", "6.0.1", "6.0.2"]

    with patch("simtools.application_control.db_handler.DatabaseHandler", return_value=mock_db):
        with patch("simtools.application_control.version.version_kind", return_value="MAJOR_MINOR"):
            with patch(
                "simtools.application_control.version.resolve_version_to_latest_patch",
                return_value="6.0.2",
            ) as mock_resolve:
                _resolve_model_version_to_latest_patch(args_dict, logger)

                mock_resolve.assert_called_once_with("6.0", ["6.0.0", "6.0.1", "6.0.2"])
                assert args_dict["model_version"] == "6.0.2"


def test_resolve_model_version_to_latest_patch_list_of_versions():
    """Test _resolve_model_version_to_latest_patch with list of versions."""

    args_dict = {"model_version": ["6.0", "6.1"]}
    logger = logging.getLogger("test")

    mock_db = MagicMock()
    mock_db.get_model_versions.return_value = ["6.0.0", "6.0.1", "6.0.2", "6.1.0", "6.1.1"]

    with patch("simtools.application_control.db_handler.DatabaseHandler", return_value=mock_db):
        with patch("simtools.application_control.version.version_kind", return_value="MAJOR_MINOR"):
            with patch(
                "simtools.application_control.version.resolve_version_to_latest_patch",
                side_effect=["6.0.2", "6.1.1"],
            ) as mock_resolve:
                _resolve_model_version_to_latest_patch(args_dict, logger)

                assert mock_resolve.call_count == 2
                assert args_dict["model_version"] == ["6.0.2", "6.1.1"]


def test_resolve_model_version_to_latest_patch_list_with_full_versions():
    """Test _resolve_model_version_to_latest_patch with list containing full versions."""

    args_dict = {"model_version": ["6.0.2", "6.1"]}
    logger = logging.getLogger("test")

    mock_db = MagicMock()
    mock_db.get_model_versions.return_value = ["6.0.2", "6.1.0", "6.1.1"]

    with patch("simtools.application_control.db_handler.DatabaseHandler", return_value=mock_db):
        with patch(
            "simtools.application_control.version.version_kind",
            side_effect=["major.minor.patch", "MAJOR_MINOR"],
        ):
            with patch(
                "simtools.application_control.version.MAJOR_MINOR_PATCH", "major.minor.patch"
            ):
                with patch(
                    "simtools.application_control.version.resolve_version_to_latest_patch",
                    return_value="6.1.1",
                ) as mock_resolve:
                    _resolve_model_version_to_latest_patch(args_dict, logger)

                    mock_resolve.assert_called_once_with("6.1", ["6.0.2", "6.1.0", "6.1.1"])
                    assert args_dict["model_version"] == ["6.0.2", "6.1.1"]


def test_resolve_model_version_to_latest_patch_empty_list():
    """Test _resolve_model_version_to_latest_patch with empty list."""

    args_dict = {"model_version": []}
    logger = logging.getLogger("test")

    _resolve_model_version_to_latest_patch(args_dict, logger)

    assert args_dict["model_version"] == []


def test_resolve_model_version_to_latest_patch_db_exception():
    """Test _resolve_model_version_to_latest_patch when database raises exception."""

    args_dict = {"model_version": "6.0"}
    logger = logging.getLogger("test")

    with patch(
        "simtools.application_control.db_handler.DatabaseHandler",
        side_effect=OSError("Database connection failed"),
    ):
        with patch("simtools.application_control.version.version_kind", return_value="MAJOR_MINOR"):
            _resolve_model_version_to_latest_patch(args_dict, logger)

            assert args_dict["model_version"] == "6.0"


def test_resolve_model_version_to_latest_patch_list_with_db_exception():
    """Test _resolve_model_version_to_latest_patch with list when database raises exception."""

    args_dict = {"model_version": ["6.0", "6.1"]}
    logger = logging.getLogger("test")

    with patch(
        "simtools.application_control.db_handler.DatabaseHandler",
        side_effect=OSError("Database connection failed"),
    ):
        with patch("simtools.application_control.version.version_kind", return_value="MAJOR_MINOR"):
            _resolve_model_version_to_latest_patch(args_dict, logger)

            assert args_dict["model_version"] == ["6.0", "6.1"]


def test_resolve_model_version_to_latest_patch_list_mixed_with_exception():
    """Test _resolve_model_version_to_latest_patch with list where one version fails."""

    args_dict = {"model_version": ["6.0", "6.1"]}
    logger = logging.getLogger("test")

    mock_db = MagicMock()
    mock_db.get_model_versions.return_value = ["6.0.0", "6.0.1"]

    with patch("simtools.application_control.db_handler.DatabaseHandler", return_value=mock_db):
        with patch("simtools.application_control.version.version_kind", return_value="MAJOR_MINOR"):
            with patch(
                "simtools.application_control.version.resolve_version_to_latest_patch",
                side_effect=["6.0.1", ValueError("Version not found")],
            ):
                _resolve_model_version_to_latest_patch(args_dict, logger)

                assert args_dict["model_version"] == ["6.0.1", "6.1"]


def test_version_info_with_build_options():
    """Test _version_info with available build options."""
    args_dict = {"run_time": "test_runtime"}
    logger = logging.getLogger("test")
    mock_io_handler = MagicMock()

    with patch("simtools.application_control.dependencies.get_build_options") as mock_build:
        with patch(
            "simtools.application_control.dependencies.get_database_version_or_name"
        ) as mock_db:
            with patch("simtools.application_control.version.__version__", "1.0.0"):
                mock_build.return_value = {
                    "corsika_version": "7.7500",
                    "simtel_version": "2021-09-01",
                }
                mock_db.side_effect = ["test_db", "1.0.0"]

                _version_info(args_dict, mock_io_handler, logger)

                mock_build.assert_called_once_with("test_runtime")


def test_version_info_no_build_options():
    """Test _version_info when build options file not found."""
    args_dict = {}
    logger = logging.getLogger("test")
    mock_io_handler = MagicMock()

    with patch(
        "simtools.application_control.dependencies.get_build_options",
        side_effect=FileNotFoundError("Build options not found"),
    ):
        _version_info(args_dict, mock_io_handler, logger)

        mock_io_handler.get_output_file.assert_not_called()


def test_version_info_export_build_info_with_io_handler():
    """Test _version_info exports build info when io_handler is available."""
    args_dict = {"run_time": "test_runtime", "export_build_info": "build_info.json"}
    logger = logging.getLogger("test")
    mock_io_handler = MagicMock()
    mock_io_handler.get_output_file.return_value = "/output/build_info.json"

    with patch("simtools.application_control.dependencies.get_build_options") as mock_build:
        with patch("simtools.application_control.dependencies.get_database_version_or_name"):
            with patch(
                "simtools.application_control.dependencies.export_build_info"
            ) as mock_export:
                with patch("simtools.application_control.version.__version__", "1.0.0"):
                    mock_build.return_value = {"corsika_version": "7.7500"}

                    _version_info(args_dict, mock_io_handler, logger)

                    mock_io_handler.get_output_file.assert_called_once_with("build_info.json")
                    mock_export.assert_called_once_with("/output/build_info.json", "test_runtime")


def test_version_info_export_build_info_without_io_handler():
    """Test _version_info exports build info using file path when io_handler is None."""
    args_dict = {"run_time": "test_runtime", "export_build_info": "/output/build_info.json"}
    logger = logging.getLogger("test")

    with patch("simtools.application_control.dependencies.get_build_options") as mock_build:
        with patch("simtools.application_control.dependencies.get_database_version_or_name"):
            with patch(
                "simtools.application_control.dependencies.export_build_info"
            ) as mock_export:
                with patch("simtools.application_control.version.__version__", "1.0.0"):
                    mock_build.return_value = {"corsika_version": "7.7500"}

                    _version_info(args_dict, None, logger)

                    mock_export.assert_called_once_with("/output/build_info.json", "test_runtime")


def test_version_info_no_export_build_info():
    """Test _version_info when export_build_info is not set."""
    args_dict = {"run_time": "test_runtime"}
    logger = logging.getLogger("test")
    mock_io_handler = MagicMock()

    with patch("simtools.application_control.dependencies.get_build_options") as mock_build:
        with patch("simtools.application_control.dependencies.get_database_version_or_name"):
            with patch(
                "simtools.application_control.dependencies.export_build_info"
            ) as mock_export:
                with patch("simtools.application_control.version.__version__", "1.0.0"):
                    mock_build.return_value = {"corsika_version": "7.7500"}

                    _version_info(args_dict, mock_io_handler, logger)

                    mock_export.assert_not_called()
