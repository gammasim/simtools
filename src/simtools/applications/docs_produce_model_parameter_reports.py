#!/usr/bin/python3

r"""
Produce a model parameter report per array element.

The markdown reports include detailed information on each parameter,
comparing their values over various model versions.
Currently only implemented for telescopes.
"""

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

    return config.initialize(db_config=True, simulation_model=["site", "telescope"])


def main():  # noqa: D103
    label_name = "reports"
    args, db_config = _parse(label_name)
    io_handler_instance = io_handler.IOHandler()
    output_path = io_handler_instance.get_output_directory(label=label_name, sub_dir="parameters")

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args["log_level"]))

    ReadParameters(
        db_config,
        args,
        output_path,
    ).produce_model_parameter_reports()

    logger.info(
        f"Markdown report generated for {args['site']} Telescope {args['telescope']}: {output_path}"
    )


if __name__ == "__main__":
    main()
