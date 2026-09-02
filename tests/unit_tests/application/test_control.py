"""Unit tests for application runtime control."""

import logging
import os
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from simtools.application.control import (
    _initialize_runtime,
    _resolve_model_version_to_latest_patch,
    _version_info,
    get_log_file,
    setup_logging,
)
from simtools.settings import config


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
    app_context = _initialize_runtime(mock_args_dict, mock_db_config, setup_io_handler=False)

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
            (
                "Setting environment variables: {{"
                "'SIMTOOLS_DB_API_PW': '{secret}', "
                "'SIMTOOLS_DB_API_USER': 'api', "
                "'SIMTOOLS_DB_SERVER': 'simtools-mongodb', "
                "'USER': 'test_user'}}"
            ),
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
            f"Environment: {'PASS' + 'WORD'}=secret123 USER=admin",
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
    app_context, handler = redact_test_setup
    _reset_stream(handler)

    app_context.logger.debug(log_message)
    output = _read_stream(handler)

    assert "***REDACTED***" in output
    for secret in secret_values:
        assert secret not in output
    for non_secret in non_secret_values:
        assert non_secret in output


def test_initialize_runtime_basic():
    mock_args_dict = {"log_level": "info", "test": True}
    mock_db_config = {"host": "localhost"}
    app_context = _initialize_runtime(mock_args_dict, mock_db_config)

    assert app_context.args == mock_args_dict
    assert app_context.db_config == mock_db_config
    assert isinstance(app_context.logger, logging.Logger)
    assert app_context.io_handler is not None

    assert app_context.logger.level == logging.INFO


def test_initialize_runtime_without_io_handler():
    """Test application runtime startup without IOHandler."""
    mock_args_dict = {"log_level": "debug"}
    mock_db_config = {}
    app_context = _initialize_runtime(mock_args_dict, mock_db_config, setup_io_handler=False)

    # Verify returned values
    assert app_context.args == mock_args_dict
    assert app_context.db_config == mock_db_config
    assert isinstance(app_context.logger, logging.Logger)
    assert app_context.io_handler is None

    # Verify logger level was set to debug
    assert app_context.logger.level == logging.DEBUG


def test_initialize_runtime_without_model_reader(mocker):
    """Write-only applications do not require a checked-out model repository."""
    model_reader = mocker.patch("simtools.application.control.create_model_reader")
    app_context = _initialize_runtime(
        {"log_level": "info"}, {}, setup_io_handler=False, initialize_model_reader=False
    )

    model_reader.assert_not_called()
    assert app_context.model_reader is None


def test_initialize_runtime_without_resolving_sim_software_executables():
    """Test runtime startup forwards executable-resolution flag to settings load."""
    mock_args_dict = {"log_level": "info"}
    mock_db_config = {}
    with patch("simtools.application.control.config.load") as mock_load:
        _initialize_runtime(
            mock_args_dict,
            mock_db_config,
            setup_io_handler=False,
            resolve_sim_software_executables=False,
        )

    mock_load.assert_called_once_with(
        mock_args_dict,
        mock_db_config,
        resolve_sim_software_executables=False,
    )


def test_initialize_runtime_validates_simulation_dependencies_after_loading():
    """Dependency validation runs after settings load and before logging."""
    mock_args_dict = {"log_level": "info", "simulation_software": "corsika"}
    mock_db_config = {}
    with (
        patch("simtools.application.control.config.load") as mock_load,
        patch(
            "simtools.application.control.dependencies.validate_simulation_dependencies"
        ) as validate,
        patch("simtools.application.control.setup_logging") as setup_logging,
    ):
        _initialize_runtime(
            mock_args_dict,
            mock_db_config,
            setup_io_handler=False,
            validate_simulation_dependencies=True,
        )

    mock_load.assert_called_once()
    validate.assert_called_once_with("corsika")
    setup_logging.assert_called_once()


def test_initialize_runtime_defers_dependency_validation_for_remote_jobs():
    """Remote grid controllers leave simulation checks to their backend workers."""
    mock_args_dict = {
        "log_level": "info",
        "simulation_software": "corsika",
        "_defer_simulation_dependency_validation": True,
    }
    mock_db_config = {}
    with (
        patch("simtools.application.control.config.load"),
        patch(
            "simtools.application.control.dependencies.validate_simulation_dependencies"
        ) as validate,
    ):
        _initialize_runtime(
            mock_args_dict,
            mock_db_config,
            setup_io_handler=False,
            validate_simulation_dependencies=True,
        )

    validate.assert_not_called()


def test_initialize_runtime_stops_when_dependencies_are_unavailable():
    """Dependency failures prevent the rest of application startup."""
    mock_args_dict = {"log_level": "info", "simulation_software": "corsika"}
    mock_db_config = {}
    with (
        patch("simtools.application.control.config.load"),
        patch(
            "simtools.application.control.dependencies.validate_simulation_dependencies",
            side_effect=ValueError("missing dependency"),
        ),
        patch("simtools.application.control.setup_logging") as setup_logging,
    ):
        with pytest.raises(ValueError, match="missing dependency"):
            _initialize_runtime(
                mock_args_dict,
                mock_db_config,
                setup_io_handler=False,
                validate_simulation_dependencies=True,
            )

    setup_logging.assert_not_called()


def test_initialize_runtime_prepares_runtime_environment_from_cli():
    mock_args_dict = {
        "log_level": "info",
        "runtime_environment_file": Path("runtime.yml"),
        "ignore_runtime_environment": False,
    }
    mock_db_config = {}
    with patch(
        "simtools.application.control.prepare_runtime_environment",
        return_value=({"image": "test-image"}, ["podman", "run"]),
    ) as mock_prepare:
        app_context = _initialize_runtime(mock_args_dict, mock_db_config, setup_io_handler=False)

    mock_prepare.assert_called_once_with(Path("runtime.yml"))
    assert app_context.run_time == ["podman", "run"]
    assert app_context.args["runtime_environment"] == {"image": "test-image"}
    assert app_context.args["run_time"] == ["podman", "run"]


def test_initialize_runtime_runtime_environment_ignored_from_cli():
    mock_args_dict = {
        "log_level": "info",
        "runtime_environment_file": Path("runtime.yml"),
        "ignore_runtime_environment": True,
    }
    mock_db_config = {}
    with patch("simtools.application.control.prepare_runtime_environment") as mock_prepare:
        app_context = _initialize_runtime(mock_args_dict, mock_db_config, setup_io_handler=False)

    mock_prepare.assert_not_called()
    assert app_context.run_time is None
    assert "runtime_environment" not in app_context.args
    assert "run_time" not in app_context.args


def test_resolve_model_version_to_latest_patch_full_version():

    args_dict = {"model_version": "6.0.1"}
    logger = logging.getLogger("test")

    model_reader = MagicMock()
    with patch("simtools.application.control.version.version_kind") as mock_version_kind:
        with patch("simtools.application.control.version.MAJOR_MINOR_PATCH", "major.minor.patch"):
            mock_version_kind.return_value = "major.minor.patch"
            _resolve_model_version_to_latest_patch(args_dict, logger, model_reader)

    assert args_dict["model_version"] == "6.0.1"


def test_resolve_model_version_to_latest_patch_resolves_to_latest():

    args_dict = {"model_version": "6.0"}
    logger = logging.getLogger("test")

    mock_db = MagicMock()
    mock_db.get_model_versions.return_value = ["6.0.0", "6.0.1", "6.0.2"]

    with patch("simtools.application.control.version.version_kind", return_value="MAJOR_MINOR"):
        with patch(
            "simtools.application.control.version.resolve_version_to_latest_patch",
            return_value="6.0.2",
        ) as mock_resolve:
            _resolve_model_version_to_latest_patch(args_dict, logger, mock_db)

            mock_resolve.assert_called_once_with("6.0", ["6.0.0", "6.0.1", "6.0.2"])
            assert args_dict["model_version"] == "6.0.2"


def test_resolve_model_version_to_latest_patch_list_of_versions():

    args_dict = {"model_version": ["6.0", "6.1"]}
    logger = logging.getLogger("test")

    mock_db = MagicMock()
    mock_db.get_model_versions.return_value = ["6.0.0", "6.0.1", "6.0.2", "6.1.0", "6.1.1"]

    with patch("simtools.application.control.version.version_kind", return_value="MAJOR_MINOR"):
        with patch(
            "simtools.application.control.version.resolve_version_to_latest_patch",
            side_effect=["6.0.2", "6.1.1"],
        ) as mock_resolve:
            _resolve_model_version_to_latest_patch(args_dict, logger, mock_db)

            assert mock_resolve.call_count == 2
            assert args_dict["model_version"] == ["6.0.2", "6.1.1"]


def test_resolve_model_version_to_latest_patch_list_with_full_versions():

    args_dict = {"model_version": ["6.0.2", "6.1"]}
    logger = logging.getLogger("test")

    mock_db = MagicMock()
    mock_db.get_model_versions.return_value = ["6.0.2", "6.1.0", "6.1.1"]

    with patch(
        "simtools.application.control.version.version_kind",
        side_effect=["major.minor.patch", "MAJOR_MINOR"],
    ):
        with patch("simtools.application.control.version.MAJOR_MINOR_PATCH", "major.minor.patch"):
            with patch(
                "simtools.application.control.version.resolve_version_to_latest_patch",
                return_value="6.1.1",
            ) as mock_resolve:
                _resolve_model_version_to_latest_patch(args_dict, logger, mock_db)

                mock_resolve.assert_called_once_with("6.1", ["6.0.2", "6.1.0", "6.1.1"])
                assert args_dict["model_version"] == ["6.0.2", "6.1.1"]


def test_resolve_model_version_to_latest_patch_db_exception():

    args_dict = {"model_version": "6.0"}
    logger = logging.getLogger("test")

    model_reader = MagicMock()
    model_reader.get_model_versions.side_effect = OSError("Database connection failed")
    with patch("simtools.application.control.version.version_kind", return_value="MAJOR_MINOR"):
        _resolve_model_version_to_latest_patch(args_dict, logger, model_reader)

        assert args_dict["model_version"] == "6.0"


def test_resolve_model_version_to_latest_patch_list_mixed_with_exception():

    args_dict = {"model_version": ["6.0", "6.1"]}
    logger = logging.getLogger("test")

    mock_db = MagicMock()
    mock_db.get_model_versions.return_value = ["6.0.0", "6.0.1"]

    with patch("simtools.application.control.version.version_kind", return_value="MAJOR_MINOR"):
        with patch(
            "simtools.application.control.version.resolve_version_to_latest_patch",
            side_effect=["6.0.1", ValueError("Version not found")],
        ):
            _resolve_model_version_to_latest_patch(args_dict, logger, mock_db)

            assert args_dict["model_version"] == ["6.0.1", "6.1"]


def test_version_info_export_build_info_with_io_handler():
    args_dict = {"run_time": "test_runtime", "export_build_info": "build_info.json"}
    logger = logging.getLogger("test")
    mock_io_handler = MagicMock()
    mock_io_handler.get_output_file.return_value = "/output/build_info.json"

    with patch("simtools.application.control.dependencies.get_build_options") as mock_build:
        with patch("simtools.application.control.dependencies.get_database_tag_or_name"):
            with patch(
                "simtools.application.control.dependencies.export_build_info"
            ) as mock_export:
                with patch("simtools.application.control.version.__version__", "1.0.0"):
                    mock_build.return_value = {"corsika_build_id": "7.7500"}

                    _version_info(args_dict, mock_io_handler, logger)

                    mock_io_handler.get_output_file.assert_called_once_with("build_info.json")
                    mock_export.assert_called_once_with("/output/build_info.json", "test_runtime")


def test_version_info_export_build_info_without_io_handler():
    args_dict = {"run_time": "test_runtime", "export_build_info": "/output/build_info.json"}
    logger = logging.getLogger("test")

    with patch("simtools.application.control.dependencies.get_build_options") as mock_build:
        with patch("simtools.application.control.dependencies.get_database_tag_or_name"):
            with patch(
                "simtools.application.control.dependencies.export_build_info"
            ) as mock_export:
                with patch("simtools.application.control.version.__version__", "1.0.0"):
                    mock_build.return_value = {"corsika_build_id": "7.7500"}

                    _version_info(args_dict, None, logger)

                    mock_export.assert_called_once_with("/output/build_info.json", "test_runtime")


def test_get_log_file_explicit_file():
    args_dict = {
        "log_file": "/path/to/custom.log",
        "application_label": "ignored",
    }
    result = get_log_file(args_dict)
    assert result == "/path/to/custom.log"


def test_get_log_file_disabled():
    args_dict = {
        "disable_log_file": True,
        "application_label": "test_app",
        "output_path": "output/test",
    }
    result = get_log_file(args_dict)
    assert result is None


def test_get_log_file_with_output_path(tmp_test_directory):
    output_path = Path(str(tmp_test_directory)) / "new_dir" / "nested"
    args_dict = {"application_label": "test_app", "output_path": str(output_path)}
    result = get_log_file(args_dict)

    assert isinstance(result, Path)
    assert result.parent == output_path
    assert result.name.startswith("test_app_")
    assert result.name.endswith(".log")
    assert output_path.exists()


def test_get_log_file_with_log_file_path_preferred_over_output_path(tmp_test_directory):
    tmp_path = Path(tmp_test_directory)
    output_path = tmp_path / "output"
    log_path = tmp_path / "logs"
    args_dict = {
        "application_label": "test_app",
        "output_path": str(output_path),
        "log_file_path": str(log_path),
    }
    result = get_log_file(args_dict)

    assert isinstance(result, Path)
    assert result.parent == log_path
    assert result.name.startswith("test_app_")
    assert result.name.endswith(".log")
    assert log_path.exists()


def test_setup_logging_with_logger_name():
    logger = setup_logging(logger_name="test_logger")
    assert logger.name == "test_logger"


def test_setup_logging_with_file_handler(tmp_test_directory):
    log_file = Path(str(tmp_test_directory)) / "test.log"
    logger = setup_logging(
        log_level="INFO", log_file=str(log_file), logger_name="test_file_handler"
    )
    try:
        logger.info("Test message")

        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) > 0
        assert file_handlers[0].baseFilename == str(log_file)
        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content
        assert config.activity_id in content
    finally:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
