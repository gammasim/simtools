#!/usr/bin/python3

r"""
Generate markdown reports for relevant model parameters.

The generated reports include detailed information on each parameter,
such as the parameter name, value, unit, description, and short description.
The reports are then uploaded as GitLab Pages using GitLab's CI workflow.
"""

import logging
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
    """Generate a markdown file to report the parameter values."""
    io_handler_instance = io_handler.IOHandler()
    output_path = io_handler_instance.get_output_directory(output_folder)
    output_filename = f'{output_path}/v{args_dict["model_version"]}_{args_dict["telescope"]}.md'

    # Start writing the Markdown file
    with open(output_filename, "w", encoding="utf-8") as file:

        # Write the section header to specify the telescope
        file.write(f"# {args_dict['telescope']}\n\n")

        for item in data:
            file.write(f"## **{item[0]}**\n")
            file.write(f"**Value:** {item[1]}\n")
            file.write(f"**Short Description:** {item[2]}\n")
            file.write("-------------------------------------------------------\n")
            file.write("\n")


def main():  # noqa: D103
    label_name = Path(__file__).stem
    args, db_config = _parse(label_name)

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args["log_level"]))

    telescope_model = TelescopeModel(
        site=args["site"],
        telescope_name=args["telescope"],
        model_version=args["model_version"],
        label=label_name,
        mongo_db_config=db_config,
    )

    parameter_data = ReadParameters(telescope_model).get_telescope_parameter_data()

    generate_markdown_report(label_name, args, parameter_data)

    logger.info(
        f"Markdown report generated for {args['site']}"
        f" Telescope {args['telescope']} (v{args['model_version']})"
    )


if __name__ == "__main__":
    main()
