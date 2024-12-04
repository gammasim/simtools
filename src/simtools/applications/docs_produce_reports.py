#!/usr/bin/python3

r"""
Generate markdown reports for relevant model parameters.

The generated reports include detailed information on each parameter,
such as the parameter name, value, unit, description, and short description.
The reports are then uploaded as GitLab Pages using GitLab's CI workflow.
"""

import logging
import subprocess
from itertools import groupby
from operator import itemgetter
from pathlib import Path

from simtools.configuration import configurator
from simtools.io_operations import io_handler
from simtools.model.telescope_model import TelescopeModel
from simtools.reporting.docs_read_parameters import ReadParameters
from simtools.utils import general as gen


def _parse(label):
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=label,
        description=("Generate a markdown report for model parameters."),
    )

    return config.initialize(db_config=True, simulation_model=["telescope"])


def generate_markdown_report(output_folder, args_dict, data):
    """
    Generate a markdown file to report the parameter values.

    Parameters
    ----------
    output_folder : str
        The folder where the markdown file will be saved.
    args_dict : dict
        Configuration arguments including model version and telescope name.
    data : list
        Parameter data in the format:
        [class, parameter_name, value, short_description]
    """
    # Sort data by class to prepare for grouping
    data.sort(key=itemgetter(0))  # Sort by the first element (class)

    output_filename = f'{output_folder}/{args_dict["telescope"]}.md'

    # Start writing the Markdown file
    with Path(output_filename).open("w", encoding="utf-8") as file:
        # Group by class and write sections
        for class_name, group in groupby(data, key=itemgetter(0)):
            file.write(f"# {class_name}\n\n")

            # Write table header and separator row
            # file.write("| Parameter Name | Values | Short Description |\n\n")
            # file.write("|----------------|--------|-------------------|\n\n")

            # Write table rows
            for _, parameter_name, value, short_description in group:
                file.write(f"**Parameter**: {parameter_name}\n\n")
                file.write(f"**Value**: {value}\n\n")
                file.write(f"**Short description**: {short_description}\n\n")
                file.write("--------------------------\n\n")
                file.write("\n\n")


def main():  # noqa: D103
    label_name = Path(__file__).stem
    args, db_config = _parse(label_name)

    io_handler_instance = io_handler.IOHandler()
    output_path = io_handler_instance.get_output_directory(label_name)
    subprocess.run(["mkdir", "-p", f"{output_path}/v{args['model_version']}"], check=True)

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args["log_level"]))

    telescope_model = TelescopeModel(
        site=args["site"],
        telescope_name=args["telescope"],
        model_version=args["model_version"],
        label=label_name,
        mongo_db_config=db_config,
    )

    parameter_data = ReadParameters(
        telescope_model, f"{output_path}/v{args['model_version']}"
    ).get_telescope_parameter_data()

    generate_markdown_report(f"{output_path}/v{args['model_version']}", args, parameter_data)

    logger.info(
        f"Markdown report generated for {args['site']}"
        f" Telescope {args['telescope']} (v{args['model_version']})"
    )


if __name__ == "__main__":
    main()
