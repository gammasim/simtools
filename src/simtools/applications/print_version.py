#!/usr/bin/python3
"""
Print the versions of the simtools software.

The versions of simtools, the DB, sim_telarray, and CORSIKA are printed.

"""

import logging
from pathlib import Path

from simtools import dependencies, version
from simtools.configuration import configurator
from simtools.utils import general as gen


def _parse(label, description, usage):
    """
    Parse command line configuration.

    No command line arguments are required for this application,
    but the configurator is called to set up the DB connection and
    the structure with _parse is kept from other applications for consistency.

    Parameters
    ----------
    label : str
        Label describing the application.
    description : str
        Description of the application.
    usage : str
        Example on how to use the application.

    Returns
    -------
    CommandLineParser
        Command line parser object.
    """
    config = configurator.Configurator(label=label, description=description, usage=usage)

    return config.initialize(db_config=True, require_command_line=False)


def main():
    """Print the versions of the simtools software."""
    args_dict, db_config = _parse(
        Path(__file__).stem,
        description="Print the versions of simtools, the DB, sim_telarray and CORSIKA.",
        usage="simtools-print-version",
    )
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    version_string = dependencies.get_version_string(db_config)

    print()
    print(f"simtools version: {version.__version__}")
    print(version_string)


if __name__ == "__main__":
    main()
