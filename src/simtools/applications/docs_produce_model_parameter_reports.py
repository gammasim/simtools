#!/usr/bin/python3

r"""
Produce a model parameter report per array element.

The markdown reports include detailed information on each parameter,
comparing their values over various model versions.
Currently only implemented for telescopes.
"""

import logging

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
        "--all_sites", action="store_true", help="Produce reports for all sites."
    )

    config.parser.add_argument(
        "--output_dir",
        type=str,
        help="Output subdirectory name within the main output directory.",
    )

    return config.initialize(db_config=True, simulation_model=["site", "telescope"])


def main():  # noqa: D103
    label_name = "reports"
    args, db_config = _parse(label_name)
    io_handler_instance = io_handler.IOHandler()
    output_path = io_handler_instance.get_output_directory()

    # Create subdirectory if output_dir is specified
    if args.get("output_dir"):
        output_path = output_path / args["output_dir"]
        output_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args["log_level"]))

    if any([args.get("all_telescopes"), args.get("all_sites")]):
        ReportGenerator(
            db_config,
            args,
            output_path,
        ).auto_generate_parameter_reports()

    else:
        ReadParameters(
            db_config,
            args,
            output_path,
        ).produce_model_parameter_reports()

        logger.info(
            f"Markdown report generated for {args['site']}"
            f"Telescope {args['telescope']}: {output_path}"
        )


if __name__ == "__main__":
    main()
