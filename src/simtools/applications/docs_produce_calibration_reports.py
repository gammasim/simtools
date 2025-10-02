#!/usr/bin/python3

r"""Produces a markdown file for calibration reports."""

from simtools.application_control import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.reporting.docs_read_parameters import ReadParameters


def _parse():
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=get_application_label(__file__),
        description=("Produce a markdown report for calibration parameters."),
    )

    return config.initialize(
        db_config=True,
        simulation_model=["model_version"],
    )


def main():
    """Produce a markdown file for calibration reports."""
    app_context = startup_application(_parse)

    output_path = app_context.io_handler.get_output_directory(
        f"{app_context.args.get('model_version')}"
    )

    read_parameters = ReadParameters(
        db_config=app_context.db_config, args=app_context.args, output_path=output_path
    )
    read_parameters.produce_calibration_reports()

    app_context.logger.info(
        f"Calibration reports for model version {app_context.args.get('model_version')} "
        "produced successfully."
    )
    app_context.logger.info(f"Output path: {output_path}")


if __name__ == "__main__":
    main()
