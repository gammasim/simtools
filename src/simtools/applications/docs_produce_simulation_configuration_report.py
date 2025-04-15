#!/usr/bin/python3

r"""Produces a markdown file for a given simulation configuration."""

import logging

from simtools.configuration import configurator
from simtools.io_operations import io_handler
from simtools.reporting.docs_read_parameters import ReadParameters
from simtools.utils import general as gen


def _parse(label):
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=label,
        description=("Produce a markdown report for model parameters."),
    )

    return config.initialize(
        db_config=True,
        simulation_model=["site", "telescope", "model_version"],
        simulation_configuration=["software"],
    )


def main():  # noqa: D103
    label_name = "reports"
    args, db_config = _parse(label_name)

    io_handler_instance = io_handler.IOHandler()
    output_path = io_handler_instance.get_output_directory(label=label_name, sub_dir="productions")

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args["log_level"]))

    read_parameters = ReadParameters(db_config=db_config, args=args, output_path=output_path)

    read_parameters.produce_simulation_configuration_report()


if __name__ == "__main__":
    main()
