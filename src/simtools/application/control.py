"""Runtime control shared by simtools applications."""

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from astropy.utils import iers

import simtools.utils.general as gen
from simtools import dependencies, version
from simtools.db import db_handler
from simtools.io import io_handler
from simtools.runners.simtools_runner import prepare_runtime_environment
from simtools.settings import config

SECRET_ENV_VAR_NAMES = ["SIMTOOLS_DB_API_PW"]
SECRET_KEY_PATTERNS = [
    r"(?:password|passwd|pwd|secret|token|api[_-]?key|auth)",
]


def _configure_iers_from_env():
    """
    Configure Astropy IERS behavior based on environment variables.

    This disables network access for IERS tables when running in offline mode.

    Controlled via:
        SIMTOOLS_OFFLINE_IERS=1
    """
    if os.getenv("SIMTOOLS_OFFLINE_IERS") != "1":
        return

    if iers is None:
        return  # nothing to configure

    iers.conf.auto_download = False
    iers.conf.use_network = False
    iers.conf.iers_degraded_accuracy = "warn"


def setup_logging(logger_name=None, log_level="INFO", log_file=None):
    """
    Set up logging configuration.

    Parameters
    ----------
    logger_name : str, optional
        Name for the logger. If None, uses the root logger. Default is None.
    log_level : str, optional
        Logging level as a string (e.g., "DEBUG", "INFO"). Default is "INFO".
    log_file : str or pathlib.Path, optional
        Path to a log file. If provided, a file handler is added and log messages
        are written to this file. If None, logging to file is disabled. Default is None.
    """
    if logger_name:
        logger = logging.getLogger(logger_name)
    else:
        logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(log_level))

    if logger.hasHandlers():
        for handler in logger.handlers[:]:
            handler.close()
        logger.handlers.clear()

    redact_filter = RedactFilter()

    console_format = logging.Formatter("%(levelname)s: %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_format)
    console_handler.addFilter(redact_filter)
    logger.addHandler(console_handler)

    # 3. File Handler
    if log_file:
        log_file_path = Path(log_file)
        if log_file_path.parent:
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
        file_format = logging.Formatter(
            f"{config.activity_id} - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(file_format)
        file_handler.addFilter(redact_filter)
        logger.addHandler(file_handler)
        logger.info(f"Log messages will be written to: {log_file_path}")

    return logger


def get_log_file(args_dict):
    """
    Get log file path.

    Generate log file path if needed from application name and application ID.

    Returns
    -------
    Path, str or None
        Log file path, or None if no logging to file is configured.
    """
    if args_dict.get("disable_log_file"):
        return None
    if args_dict.get("log_file") is not None:
        return args_dict["log_file"]
    log_file_path = args_dict.get("log_file_path") or args_dict.get("output_path")
    if args_dict.get("application_label") is None or log_file_path is None:
        return None

    log_file = f"{args_dict['application_label']}_{config.activity_id}.log"
    Path(log_file_path).mkdir(parents=True, exist_ok=True)
    return Path(log_file_path) / log_file


class RedactFilter(logging.Filter):
    """
    Filter to redact sensitive information from log messages.

    This filter dynamically retrieves secret values from environment variables
    and uses pattern matching to identify and redact sensitive key-value pairs.
    """

    def filter(self, record):
        """Filter log record to redact sensitive information."""
        msg = record.getMessage()

        for env_var_name in SECRET_ENV_VAR_NAMES:
            secret_value = os.getenv(env_var_name)
            if secret_value:
                # Environment-variable-based redaction is intentionally case-sensitive
                # and only guarantees removal of the exact secret value.
                msg = msg.replace(secret_value, "***REDACTED***")

        for pattern in SECRET_KEY_PATTERNS:
            # Handles: 'key': 'value', "key": "value", 'key': "value", etc.
            msg = re.sub(
                rf"(['\"][^'\"]{{0,1000}}{pattern}[^'\"]{{0,1000}}['\"])\s*:\s*"
                rf"(['\"][^'\"]{{0,1000}}['\"])",
                lambda m: f"{m.group(1)}: '***REDACTED***'",
                msg,
                flags=re.IGNORECASE,
            )
            # Match environment variable style: KEY=value (with word boundaries)
            msg = re.sub(
                rf"\b([A-Z_]*{pattern}[A-Z_]*)=([^\s,}}]+)",
                lambda m: f"{m.group(1)}=***REDACTED***",
                msg,
                flags=re.IGNORECASE,
            )

        record.msg = msg
        record.args = ()
        return True


@dataclass
class ApplicationContext:
    """Container for common application context elements."""

    args: dict
    db_config: dict
    logger: logging.Logger
    io_handler: io_handler.IOHandler | None
    run_time: list | None = None


def _initialize_runtime(
    args_dict,
    db_config,
    setup_io_handler=True,
    resolve_sim_software_executables=True,
):
    """Initialize common runtime services for parsed application configuration.

    Parameters
    ----------
    args_dict : dict
        Parsed application configuration.
    db_config : dict
        Database configuration.
    setup_io_handler : bool, optional
        Whether to initialize and return an IOHandler instance. Default is True.
    resolve_sim_software_executables : bool, optional
        Resolve simulation software executable paths during settings load.
        Set to False for applications that only orchestrate other applications.

    Returns
    -------
    ApplicationContext
        Container holding parsed arguments, database configuration, logger, and the optional
        IO handler instance.

    """
    _configure_iers_from_env()

    config.load(
        args_dict,
        db_config,
        resolve_sim_software_executables=resolve_sim_software_executables,
    )

    logger = setup_logging(log_level=args_dict["log_level"], log_file=get_log_file(args_dict))
    logger.info(
        f"simtools application {args_dict.get('application_label')}"
        f" started with activity ID {config.activity_id}"
    )

    io_handler_instance = io_handler.IOHandler() if setup_io_handler else None

    _resolve_model_version_to_latest_patch(args_dict, logger)
    _version_info(args_dict, io_handler_instance, logger)

    run_time = _prepare_runtime_environment_from_cli(args_dict)

    return ApplicationContext(
        args=args_dict,
        db_config=db_config,
        logger=logger,
        io_handler=io_handler_instance,
        run_time=run_time,
    )


def _prepare_runtime_environment_from_cli(args_dict):
    """Prepare runtime environment from CLI arguments when requested."""
    runtime_environment_file = args_dict.get("runtime_environment_file")
    if runtime_environment_file is None or args_dict.get("ignore_runtime_environment"):
        return None

    runtime_environment, run_time = prepare_runtime_environment(runtime_environment_file)
    args_dict["runtime_environment"] = runtime_environment
    args_dict["run_time"] = run_time
    return run_time


def _resolve_model_version_to_latest_patch(args_dict, logger):
    """
    Update model_version in args_dict to latest patch version if needed.

    Updated to the latest patch version requires to setup a DB connection.

    Parameters
    ----------
    args_dict : dict
        Parsed command line arguments and configuration.
    logger : logging.Logger
        Logger instance for logging information.
    """
    mv = args_dict.get("model_version")
    if not mv:
        return

    versions = mv if isinstance(mv, list) else [mv]
    kinds = [version.version_kind(v) for v in versions]
    if all(k == version.MAJOR_MINOR_PATCH for k in kinds):
        return

    try:
        db = db_handler.DatabaseHandler()
        model_versions = db.get_model_versions()
    except (ValueError, KeyError, OSError) as exc:
        logger.warning(f"Could not connect to database, using version(s) as-is. Error: {exc}")
        return

    def resolve(v, k):
        if k == version.MAJOR_MINOR_PATCH:
            return v
        try:
            latest = version.resolve_version_to_latest_patch(v, model_versions)
            logger.info(f"Resolved {v} to {latest}")
            return latest
        except (ValueError, KeyError) as exc:
            logger.warning(f"Could not resolve {v}, using as-is. Error: {exc}")
            return v

    resolved = [resolve(v, k) for v, k in zip(versions, kinds)]
    args_dict["model_version"] = resolved if isinstance(mv, list) else resolved[0]


def _version_info(args_dict, io_handler_instance, logger):
    """Print and optionally write version information."""
    try:
        build_options = dependencies.get_build_options(args_dict.get("run_time"))
    except FileNotFoundError:
        logger.debug("No build options found.")
        return

    logger.info(
        f"simtools: {version.__version__} "
        f"DB: {dependencies.get_database_version_or_name(version=False)} "
        f"{dependencies.get_database_version_or_name(version=True)} "
        f"CORSIKA: {build_options.get('corsika_version')} "
        f"sim_telarray: {build_options.get('simtel_version')}"
    )
    logger.debug(f"Build options:\n {build_options}")

    if args_dict.get("export_build_info"):
        output_path = (
            io_handler_instance.get_output_file(args_dict["export_build_info"])
            if io_handler_instance
            else args_dict["export_build_info"]
        )
        dependencies.export_build_info(output_path, args_dict.get("run_time"))
