"""Application control utilities for startup and shutdown simtools applications."""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.io import io_handler


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

    Example
    -------
    Basic usage in an application:

    .. code-block:: python

        def main():
            args_dict, db_config, logger, io_handler_instance = startup_application(_parse)

            # Application-specific code follows
            logger.info("Starting application")
            # ... rest of application logic

    Usage without IOHandler:

    .. code-block:: python

        def main():
            args_dict, db_config, logger, _ = startup_application(_parse, setup_io_handler=False)

            # Application-specific code follows
            logger.info("Starting application")
            # ... rest of application logic
    """
    args_dict, db_config = parse_function()

    if logger_name:
        logger = logging.getLogger(logger_name)
    else:
        logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    io_handler_instance = io_handler.IOHandler() if setup_io_handler else None

    return args_dict, db_config, logger, io_handler_instance


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

    Example
    -------
    .. code-block:: python

        def main():
            label = get_application_label(__file__)
            # label will be the filename without .py extension
    """
    return Path(file_path).stem
