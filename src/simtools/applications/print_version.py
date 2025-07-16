#!/usr/bin/python3
"""
Print the versions of the simtools software.

The versions of simtools, the DB, sim_telarray, and CORSIKA are printed.

"""

import json
import logging
from pathlib import Path

from simtools import dependencies, version
from simtools.configuration import configurator
from simtools.io_operations import io_handler
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

    return config.initialize(db_config=True, output=True)


def main():
    """Print the versions of the simtools software."""
    label = Path(__file__).stem
    args_dict, db_config = _parse(
        label=label,
        description="Print the versions of simtools, the DB, sim_telarray and CORSIKA.",
        usage="simtools-print-version",
    )
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))
    io_handler_instance = io_handler.IOHandler()

    version_string = dependencies.get_version_string(db_config)
    version_dict = {"simtools version": version.__version__}

    print()
    # The loop below is not necessary, there is only one entry, but it is cleaner
    for key, value in version_dict.items():  #
        print(f"{key}: {value}")
    print(version_string)

    version_list = version_string.strip().split("\n")
    for version_entry in version_list:
        key, value = version_entry.split(": ", 1)
        version_dict[key] = value

    with open(
        io_handler_instance.get_output_file(args_dict["output_file"], label=label),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(version_dict, f, indent=4)


if __name__ == "__main__":
    main()
