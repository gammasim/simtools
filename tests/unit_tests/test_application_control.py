"""Unit tests for application_control module."""

import logging
import os
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from simtools.application_control import (
    _resolve_model_version_to_latest_patch,
    _version_info,
    get_application_label,
    get_log_file,
    setup_logging,
    startup_application,
)


def _reset_stream(handler):
    """Helper to reset stream for reading output."""
    handler.stream.seek(0)
    handler.stream.truncate()


def _read_stream(handler):
    """Helper to read stream output."""
    handler.stream.seek(0)
    return handler.stream.read()


@pytest.fixture
def redact_test_setup():
    """Set up logging handler and application context for redaction testing."""
    mock_args_dict = {"log_level": "debug"}
    mock_db_config = {}
    mock_parse_function = MagicMock(return_value=(mock_args_dict, mock_db_config))
    app_context = startup_application(mock_parse_function, setup_io_handler=False)

    handler = app_context.logger.handlers[0] if app_context.logger.handlers else None

    if handler and isinstance(handler, logging.StreamHandler):
        handler.stream = StringIO()

    yield app_context, handler

    if handler:
        handler.close()
        app_context.logger.removeHandler(handler)


@pytest.mark.parametrize(
    ("log_message", "secret_value", "env_var", "non_secret_values"),
    [
        (
            "Database password is: {secret}",
            "my_secret_password_123",
            "SIMTOOLS_DB_API_PW",
            [],
        ),
        (
            "Setting environment variables: {{"
            "'SIMTOOLS_DB_API_PW': '{secret}', "
            "'SIMTOOLS_DB_API_USER': 'api', "
            "'SIMTOOLS_DB_SERVER': 'simtools-mongodb', "
            "'USER': 'test_user'}}",
            "my_secret_db_password",
            "SIMTOOLS_DB_API_PW",
            ["api", "simtools-mongodb"],
        ),
        (
            "Environment: {{'SIMTOOLS_DB_API_PW': '{secret}', 'USER': 'test'}}",
            "child_logger_secret_789",
            "SIMTOOLS_DB_API_PW",
            ["test"],
        ),
    ],
)
def test_redact_filter_env_var(
    redact_test_setup, log_message, secret_value, env_var, non_secret_values
):
    """Test that RedactFilter redacts secret values from environment variables."""
    app_context, handler = redact_test_setup

    with patch.dict(os.environ, {env_var: secret_value}, clear=False):
        _reset_stream(handler)
        app_context.logger.info(log_message.format(secret=secret_value))
        output = _read_stream(handler)

        assert "***REDACTED***" in output
        assert secret_value not in output
        for non_secret in non_secret_values:
            assert non_secret in output


@pytest.mark.parametrize(
    ("log_message", "secret_values", "non_secret_values"),
    [
        (
            'Settings: {{"api_key": "abc123xyz", "host": "localhost"}}',
            ["abc123xyz"],
            ["localhost"],
        ),
        (
            "Auth: {{'auth_token': 'xyz789', 'service': 'api'}}",
            ["xyz789"],
            ["api"],
        ),
        (
            "Environment: PASSWORD=secret123 USER=admin",
            ["secret123"],
            ["admin"],
        ),
        (
            "Variables: api_key=xyz789, host=localhost",
            ["xyz789"],
            ["localhost"],
        ),
    ],
)
def test_redact_filter_pattern_matching(
    redact_test_setup, log_message, secret_values, non_secret_values
):
    """Test that RedactFilter redacts secrets based on pattern matching."""
    app_context, handler = redact_test_setup
    _reset_stream(handler)

    app_context.logger.debug(log_message)
    output = _read_stream(handler)

    assert "***REDACTED***" in output
    for secret in secret_values:
        assert secret not in output
    for non_secret in non_secret_values:
        assert non_secret in output


def test_redact_filter_child_logger(redact_test_setup):
    """Test that RedactFilter works for child loggers."""
    _, handler = redact_test_setup
    test_password = "child_logger_secret_789"

    with patch.dict(os.environ, {"SIMTOOLS_DB_API_PW": test_password}, clear=False):
        child_logger = logging.getLogger("simtools.job_execution.job_manager")
        child_logger.setLevel(logging.DEBUG)
        _reset_stream(handler)

        child_logger.debug(
            f"Environment: {{'SIMTOOLS_DB_API_PW': '{test_password}', 'USER': 'test'}}"
        )
        output = _read_stream(handler)

        assert "***REDACTED***" in output
        assert test_password not in output


@pytest.mark.parametrize(
    ("file_path", "expected"),
    [
        ("/path/to/my_application.py", "my_application"),
        ("/another/path/test_script.py", "test_script"),
        ("simple_app.py", "simple_app"),
    ],
)
def test_get_application_label(file_path, expected):
    """Test get_application_label function."""
    result = get_application_label(file_path)
    assert result == expected


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


def test_get_log_file_explicit_file():
    """Test get_log_file when log_file is explicitly provided (takes precedence)."""
    args_dict = {
        "log_file": "/path/to/custom.log",
        "application_label": "ignored",
    }
    result = get_log_file(args_dict)
    assert result == "/path/to/custom.log"


def test_get_log_file_no_application_label():
    """Test get_log_file returns None when application_label is not set."""
    args_dict = {}
    result = get_log_file(args_dict)
    assert result is None


def test_get_log_file_with_output_path(tmp_path):
    """Test get_log_file generates path and creates directory."""
    output_path = tmp_path / "new_dir" / "nested"
    args_dict = {"application_label": "test_app", "output_path": str(output_path)}
    result = get_log_file(args_dict)

    assert isinstance(result, Path)
    assert result.parent == output_path
    assert result.name.startswith("test_app_")
    assert result.name.endswith(".log")
    assert output_path.exists()


@pytest.mark.parametrize(
    ("log_level", "expected_level"),
    [
        ("INFO", logging.INFO),
        ("DEBUG", logging.DEBUG),
        ("WARNING", logging.WARNING),
    ],
)
def test_setup_logging_log_levels(log_level, expected_level):
    """Test setup_logging with different log levels."""
    logger = setup_logging(log_level=log_level)
    assert logger.level == expected_level
    assert len(logger.handlers) > 0


def test_setup_logging_with_logger_name():
    """Test setup_logging with custom logger name."""
    logger = setup_logging(logger_name="test_logger")
    assert logger.name == "test_logger"


def test_setup_logging_with_file_handler(tmp_path):
    """Test setup_logging creates and writes to file handler."""
    log_file = tmp_path / "test.log"
    logger = setup_logging(
        log_level="INFO", log_file=str(log_file), logger_name="test_file_handler"
    )
    try:
        logger.info("Test message")

        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) > 0
        assert file_handlers[0].baseFilename == str(log_file)
        assert log_file.exists()
        assert "Test message" in log_file.read_text()
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


def test_setup_logging_handlers_have_formatters():
    """Test that setup_logging creates handlers with formatters."""
    logger = setup_logging(logger_name="test_format")

    stream_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
    assert len(stream_handlers) > 0
    assert stream_handlers[0].formatter is not None


def test_setup_logging_clears_existing_handlers():
    """Test that setup_logging clears existing handlers to avoid duplicates."""
    logger = logging.getLogger("test_clear_handlers")
    logger.addHandler(logging.StreamHandler())
    initial_handler_count = len(logger.handlers)

    setup_logging(logger_name="test_clear_handlers")

    assert len(logger.handlers) <= initial_handler_count


@pytest.mark.parametrize(
    ("test_id", "log_message", "secret_value"),
    [
        ("env", "Password is {secret}", "test_secret_123"),
        ("api", 'Config: {{"api_key": "{secret}"}}', "secret_xyz"),
    ],
)
def test_setup_logging_redaction(tmp_path, test_id, log_message, secret_value):
    """Test that setup_logging applies redaction filter."""
    log_file = tmp_path / f"test_{test_id}.log"

    with patch.dict(os.environ, {"SIMTOOLS_DB_API_PW": secret_value}, clear=False):
        logger = setup_logging(logger_name=f"test_redact_{test_id}", log_file=str(log_file))
        logger.info(log_message.format(secret=secret_value))

        content = log_file.read_text()
        assert "***REDACTED***" in content
        assert secret_value not in content
