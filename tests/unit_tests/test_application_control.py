"""Unit tests for application_control module."""

import logging
import os
from io import StringIO
from unittest.mock import MagicMock, patch

from simtools.application_control import (
    _resolve_model_version_to_latest_patch,
    _version_info,
    get_application_label,
    startup_application,
)


def test_redact_filter_env_var_value():
    """Test that RedactFilter redacts secret values from environment variables."""
    mock_args_dict = {"log_level": "debug"}
    mock_db_config = {}
    mock_parse_function = MagicMock(return_value=(mock_args_dict, mock_db_config))

    handler = logging.StreamHandler(StringIO())
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)

    test_password = "my_secret_password_123"

    try:
        with patch.dict(os.environ, {"SIMTOOLS_DB_API_PW": test_password}, clear=False):
            app_context = startup_application(mock_parse_function, setup_io_handler=False)

            stream = handler.stream
            stream.seek(0)
            stream.truncate()

            app_context.logger.info(f"Database password is: {test_password}")

            stream.seek(0)
            output = stream.read()

            assert "***REDACTED***" in output
            assert test_password not in output
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def test_redact_filter_simtools_password():
    """Test that SIMTOOLS_DB_API_PW password values are redacted from logs."""
    mock_args_dict = {"log_level": "debug"}
    mock_db_config = {}
    mock_parse_function = MagicMock(return_value=(mock_args_dict, mock_db_config))

    handler = logging.StreamHandler(StringIO())
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)

    test_password = "my_secret_db_password"

    try:
        with patch.dict(os.environ, {"SIMTOOLS_DB_API_PW": test_password}, clear=False):
            app_context = startup_application(mock_parse_function, setup_io_handler=False)

            stream = handler.stream
            stream.seek(0)
            stream.truncate()

            # Simulate the actual log message from job_manager that the user reported
            env_dict = {
                "SIMTOOLS_DB_API_PW": test_password,
                "SIMTOOLS_DB_API_USER": "api",
                "SIMTOOLS_DB_SERVER": "simtools-mongodb",
                "USER": "test_user",
            }
            app_context.logger.debug(f"Setting environment variables for job execution: {env_dict}")

            stream.seek(0)
            output = stream.read()

            # The password value should be redacted
            assert "***REDACTED***" in output
            assert test_password not in output
            # Other non-secret values should remain
            assert "api" in output
            assert "simtools-mongodb" in output
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def test_redact_filter_api_key_pattern():
    """Test that API keys are redacted based on pattern matching."""
    mock_args_dict = {"log_level": "debug"}
    mock_db_config = {}
    mock_parse_function = MagicMock(return_value=(mock_args_dict, mock_db_config))

    handler = logging.StreamHandler(StringIO())
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)

    try:
        app_context = startup_application(mock_parse_function, setup_io_handler=False)

        stream = handler.stream
        stream.seek(0)
        stream.truncate()

        # Test API key pattern matching in double-quoted dictionary
        app_context.logger.debug('Config: {"api_key": "abc123xyz", "host": "localhost"}')
        # Test auth token pattern matching
        app_context.logger.debug("Auth: {'auth_token': 'xyz789', 'service': 'api'}")

        stream.seek(0)
        output = stream.read()

        # Secret values should be redacted
        assert "***REDACTED***" in output
        assert "abc123xyz" not in output
        assert "xyz789" not in output
        # Non-secret values should remain
        assert "localhost" in output
        assert "api" in output
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def test_redact_filter_env_var_format():
    """Test that RedactFilter redacts environment variable assignments."""
    mock_args_dict = {"log_level": "debug"}
    mock_db_config = {}
    mock_parse_function = MagicMock(return_value=(mock_args_dict, mock_db_config))

    handler = logging.StreamHandler(StringIO())
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)

    try:
        app_context = startup_application(mock_parse_function, setup_io_handler=False)

        stream = handler.stream
        stream.seek(0)
        stream.truncate()

        app_context.logger.debug("Environment: PASSWORD=secret123 USER=admin")
        app_context.logger.debug("Variables: api_key=xyz789, host=localhost")

        stream.seek(0)
        output = stream.read()

        assert "***REDACTED***" in output
        assert "secret123" not in output
        assert "xyz789" not in output
        assert "admin" in output  # non-secret value
        assert "localhost" in output
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def test_redact_filter_child_logger():
    """Test that RedactFilter works for child loggers."""
    mock_args_dict = {"log_level": "debug"}
    mock_db_config = {}
    mock_parse_function = MagicMock(return_value=(mock_args_dict, mock_db_config))

    handler = logging.StreamHandler(StringIO())
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)

    test_password = "child_logger_secret_789"

    try:
        with patch.dict(os.environ, {"SIMTOOLS_DB_API_PW": test_password}, clear=False):
            startup_application(mock_parse_function, setup_io_handler=False)

            child_logger = logging.getLogger("simtools.job_execution.job_manager")
            child_logger.setLevel(logging.DEBUG)

            stream = handler.stream
            stream.seek(0)
            stream.truncate()

            # Test both the actual password and key-value pairs
            child_logger.debug(
                f"Environment: {{'SIMTOOLS_DB_API_PW': '{test_password}', 'USER': 'test'}}"
            )

            stream.seek(0)
            output = stream.read()

            assert "***REDACTED***" in output
            assert test_password not in output
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def test_get_application_label():
    """Test get_application_label function."""
    file_path = "/path/to/my_application.py"
    result = get_application_label(file_path)
    assert result == "my_application"

    file_path = "/another/path/test_script.py"
    result = get_application_label(file_path)
    assert result == "test_script"

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
