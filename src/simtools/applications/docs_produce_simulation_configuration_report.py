#!/usr/bin/python3

r"""Produces a markdown file for a given simulation configuration."""

from simtools.application_control import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.reporting.docs_read_parameters import ReadParameters


def _parse():
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=get_application_label(__file__),
        description=("Produce a markdown report for model parameters."),
    )

    return config.initialize(
        db_config=True,
        simulation_model=["model_version"],
        simulation_configuration=["software"],
    )


def main():
    """Produce a markdown file for a given simulation configuration."""
    args_dict, db_config, logger, _io_handler = startup_application(_parse)

    output_path = _io_handler.get_output_directory(f"{args_dict.get('model_version')}")

    read_parameters = ReadParameters(db_config=db_config, args=args_dict, output_path=output_path)

    read_parameters.produce_simulation_configuration_report()

    logger.info(
        f"Configuration reports for {args_dict.get('simulation_software')} produced successfully."
    )
    logger.info(f"Output path: {output_path}")


if __name__ == "__main__":
    main()
