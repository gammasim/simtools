"""Application control utilities for startup and shutdown simtools applications."""

import inspect
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from astropy.utils import iers

import simtools.utils.general as gen
from simtools import dependencies, version
from simtools.configuration import configurator
from simtools.db import db_handler
from simtools.io import io_handler
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
    if args_dict.get("log_file") is not None:
        return args_dict["log_file"]
    if args_dict.get("application_label") is None or args_dict.get("output_path") is None:
        return None

    log_file = f"{args_dict['application_label']}_{config.activity_id}.log"
    Path(args_dict["output_path"]).mkdir(parents=True, exist_ok=True)
    return Path(args_dict["output_path"]) / log_file


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


def build_application(
    application_path=None,
    description=None,
    add_arguments_function=None,
    initialization_kwargs=None,
    startup_kwargs=None,
    usage=None,
    epilog=None,
    parse_function=None,
):
    """
    Build and start an application using the standard simtools startup flow.

    Parameters
    ----------
    application_path : str, optional
        Application file path, typically ``__file__``.
        If not provided, it is inferred from the caller module.
    description : str, optional
        Application description shown in the CLI help (reduced to first line).
        If not provided, it is inferred from the caller module docstring.
    add_arguments_function : callable, optional
        Function receiving the application's ``CommandLineParser`` instance to register
        application-specific arguments. If not provided, ``_add_arguments`` from the
        caller module is used when available.
    initialization_kwargs : dict, optional
        Keyword arguments forwarded to ``Configurator.initialize``.
    startup_kwargs : dict, optional
        Keyword arguments forwarded to ``startup_application``.
    usage : str, optional
        CLI usage string.
    epilog : str, optional
        CLI epilog.
    parse_function : callable, optional
        Existing parser function returning ``(args_dict, db_config)``. If provided,
        ``build_application`` delegates directly to ``startup_application`` with this
        parser function.

    Returns
    -------
    ApplicationContext
        Application context returned by ``startup_application``.
    """
    initialization_kwargs = initialization_kwargs or {}
    startup_kwargs = startup_kwargs or {}

    if application_path is None or description is None or add_arguments_function is None:
        caller_globals = inspect.currentframe().f_back.f_globals
        if application_path is None:
            application_path = caller_globals.get("__file__")
        if description is None:
            description = caller_globals.get("__doc__")
        if add_arguments_function is None:
            add_arguments_function = caller_globals.get("_add_arguments")

    if application_path is None:
        raise ValueError("Missing application path; provide application_path explicitly.")
    if description is None:
        raise ValueError("Missing description; provide description explicitly.")

    if parse_function is not None:
        return startup_application(parse_function, **startup_kwargs)

    def _parse():
        config_builder = configurator.Configurator(
            label=get_application_label(application_path),
            usage=usage,
            description=get_module_description_line(description),
            epilog=epilog,
        )
        if add_arguments_function is not None:
            add_arguments_function(config_builder.parser)
        return config_builder.initialize(**initialization_kwargs)

    return startup_application(_parse, **startup_kwargs)


def startup_application(
    parse_function,
    setup_io_handler=True,
    logger_name=None,
    resolve_sim_software_executables=True,
):
    """
    Initialize common application startup tasks.

    This function handles the repetitive startup tasks common to most simtools applications:

    - Parse command line arguments and configuration
    - Set up logging with appropriate level
    - Optionally initialize IOHandler

    Parameters
    ----------
    parse_function : Callable
        Function that parses configuration and returns (args_dict, db_config) tuple.
        This should be the application's _parse() function.
    setup_io_handler : bool, optional
        Whether to initialize and return an IOHandler instance. Default is True.
    logger_name : str, optional
        Name for the logger. If None, uses the root logger. Default is None.
    resolve_sim_software_executables : bool, optional
        Resolve simulation software executable paths during settings load.
        Set to False for applications that only orchestrate other applications.

    Returns
    -------
    args_dict : dict
        Parsed command line arguments and configuration.
    db_config : dict
        Database configuration dictionary.
    logger : logging.Logger
        Configured logger instance.
    io_handler_instance : io_handler.IOHandler or None
        IOHandler instance if setup_io_handler=True, None otherwise.

    Examples
    --------
    Basic usage in an application:

    .. code-block:: python

        def main():
            app_context = startup_application(_parse)

            # Application-specific code follows
            app_context.logger.info("Starting application")
            # ... rest of application logic

    Usage without IOHandler:

    .. code-block:: python

        def main():
            app_context = startup_application(_parse, setup_io_handler=False)

            # Application-specific code follows
            app_context.logger.info("Starting application")
            # ... rest of application logic
    """
    _configure_iers_from_env()

    args_dict, db_config = parse_function()
    config.load(
        args_dict,
        db_config,
        resolve_sim_software_executables=resolve_sim_software_executables,
    )

    logger = setup_logging(logger_name, args_dict["log_level"], log_file=get_log_file(args_dict))
    logger.info(
        f"simtools application {args_dict.get('application_label')}"
        f" started with activity ID {config.activity_id}"
    )

    io_handler_instance = io_handler.IOHandler() if setup_io_handler else None

    _resolve_model_version_to_latest_patch(args_dict, logger)

    _version_info(args_dict, io_handler_instance, logger)

    return ApplicationContext(
        args=args_dict,
        db_config=db_config,
        logger=logger,
        io_handler=io_handler_instance,
    )


def get_application_label(file_path):
    """
    Get application label from file path.

    This is a convenience function to extract the application name from __file__.

    Parameters
    ----------
    file_path : str
        The __file__ variable from the calling application.

    Returns
    -------
    str
        Application label (filename without extension).

    Examples
    --------
    .. code-block:: python

        def main():
            label = get_application_label(__file__)
            # label will be the filename without .py extension
    """
    return Path(file_path).stem


def get_module_description_line(docstring):
    """Return the first non-empty line from a docstring.

    Parameters
    ----------
    docstring : str
        Module docstring (typically from __doc__).

    Returns
    -------
    str
        First non-empty line from the docstring.

    Raises
    ------
    ValueError
        If docstring is None or empty.
    """
    if not docstring:
        raise ValueError("Missing or empty docstring")

    for line in docstring.splitlines():
        if line.strip():
            return line.strip()

    raise ValueError("Empty docstring (only whitespace)")


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
