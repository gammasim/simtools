"""Application control utilities for startup and shutdown simtools applications."""

import logging
from dataclasses import dataclass
from pathlib import Path

import simtools.utils.general as gen
from simtools import version
from simtools.db import db_handler
from simtools.io import io_handler


@dataclass
class ApplicationContext:
    """Container for common application context elements."""

    args: dict
    db_config: dict
    logger: logging.Logger
    io_handler: io_handler.IOHandler | None


def startup_application(parse_function, setup_io_handler=True, logger_name=None):
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
    args_dict, db_config = parse_function()

    if logger_name:
        logger = logging.getLogger(logger_name)
    else:
        logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    io_handler_instance = io_handler.IOHandler() if setup_io_handler else None

    _resolve_model_version_to_latest_patch(args_dict, db_config, logger)

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


def _resolve_model_version_to_latest_patch(args_dict, db_config, logger):
    """
    Update model_version in args_dict to latest patch version if needed.

    Updated to the latest patch version requires to setup a DB connection.

    Parameters
    ----------
    args_dict : dict
        Parsed command line arguments and configuration.
    db_config : dict
        Database configuration dictionary.
    logger : logging.Logger
        Logger instance for logging information.
    """
    model_version = args_dict.get("model_version")
    if not model_version or not db_config:
        return
    if version.version_kind(model_version) == version.MAJOR_MINOR_PATCH:
        return

    db = db_handler.DatabaseHandler(db_config)
    latest = version.resolve_version_to_latest_patch(model_version, db.get_model_versions())
    logger.info(f"Resolved model_version {model_version} to latest patch version {latest}")
    args_dict["model_version"] = latest
