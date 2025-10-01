#!/usr/bin/python3

r"""Produces a markdown file for a given simulation configuration."""

import logging

from simtools.configuration import configurator
from simtools.io import io_handler
from simtools.reporting.docs_auto_report_generator import ReportGenerator
from simtools.utils import general as gen


def _parse(label):
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=label,
        description=("Produce a markdown report for model parameters."),
    )

    config.parser.add_argument(
        "--all_model_versions",
        action="store_true",
        help="Produce reports for all model versions.",
    )

    config.parser.add_argument(
        "--output_dir",
        type=str,
        help="Output subdirectory name within the main output directory.",
    )

    return config.initialize(
        db_config=True,
        simulation_model=["model_version"],
        simulation_configuration=["software"],
    )


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

    # Use ReportGenerator so --all_model_versions is supported consistently
    report_generator = ReportGenerator(db_config=db_config, args=args, output_path=output_path)
    report_generator.auto_generate_simulation_configuration_reports()

    logger.info(
        f"Configuration reports for {args.get('simulation_software')} produced successfully."
    )
    logger.info(f"Output path: {output_path}/{args.get('model_version')}/")


if __name__ == "__main__":
    main()
