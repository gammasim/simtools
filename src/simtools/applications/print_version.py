#!/usr/bin/python3
"""
Print the versions of the simtools software.

The versions of simtools, the DB, sim_telarray, and CORSIKA are printed.

"""

from simtools import dependencies, version
from simtools.application_control import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.io import ascii_handler


def _parse():
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=get_application_label(__file__),
        description="Print the versions of simtools, the DB, sim_telarray and CORSIKA.",
        usage="simtools-print-version",
    )
    return config.initialize(db_config=True, output=True, require_command_line=False)


def main():
    """Print the versions of the simtools software."""
    args_dict, db_config, _, _io_handler = startup_application(_parse)

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

    if not args_dict.get("output_file_from_default", False):
        ascii_handler.write_data_to_file(
            data=version_dict,
            output_file=_io_handler.get_output_file(
                args_dict.get("output_file", "simtools_version.json")
            ),
        )


if __name__ == "__main__":
    main()
