#!/usr/bin/python3

r"""
Produces a markdown file for a given array element, site, and model version.

The report includes detailed information on each parameter,
such as the parameter name, value, unit, description, and short description.
"""

import logging
from pathlib import Path

from simtools.configuration import configurator
from simtools.io import io_handler
from simtools.reporting.docs_auto_report_generator import ReportGenerator
from simtools.reporting.docs_read_parameters import ReadParameters
from simtools.utils import general as gen


def _parse(label):
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=label,
        description=("Produce a markdown report for model parameters."),
    )

    config.parser.add_argument(
        "--all_telescopes",
        action="store_true",
        help="Produce reports for all telescopes.",
    )

    config.parser.add_argument(
        "--all_model_versions",
        action="store_true",
        help="Produce reports for all model versions.",
    )

    config.parser.add_argument(
        "--all_sites", action="store_true", help="Produce reports for all sites."
    )

    config.parser.add_argument(
        "--observatory",
        action="store_true",
        help="Produce reports for an observatory at a given site.",
    )

    return config.initialize(
        db_config=True, simulation_model=["site", "telescope", "model_version"]
    )


def main():  # noqa: D103
    label_name = "reports"
    args, db_config = _parse(label_name)

    io_handler_instance = io_handler.IOHandler()
    output_path = io_handler_instance.get_output_directory()

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args["log_level"]))

    if any([args.get("all_telescopes"), args.get("all_sites"), args.get("all_model_versions")]):
        ReportGenerator(
            db_config,
            args,
            output_path,
        ).auto_generate_array_element_reports()

    else:
        model_version = args["model_version"]
        ReadParameters(
            db_config,
            args,
            Path(output_path / f"{model_version}"),
        ).produce_array_element_report()

        logger.info(
            f"Markdown report generated for {args['site']}"
            f" Telescope {args['telescope']} (v{model_version}):"
            f" {output_path}"
        )


if __name__ == "__main__":
    main()
