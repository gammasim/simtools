#!/usr/bin/python3

r"""Produces a markdown file for calibration reports."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.reporting.docs_auto_report_generator import ReportGenerator

_ARGUMENTS = (cli.ALL_MODEL_VERSIONS,)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        cli.MODEL_VERSION,
        cli.OVERWRITE_MODEL_PARAMETERS,
        *cli.PATH_ARGUMENTS,
    ),
    database=True,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    output_path = app_context.io_handler.get_output_directory()

    generator = ReportGenerator(args=app_context.args, output_path=output_path)
    generator.auto_generate_calibration_reports()

    if app_context.args.get("all_model_versions"):
        app_context.logger.info("Calibration reports for all model versions produced successfully.")
    else:
        app_context.logger.info(
            f"Calibration reports for model version {app_context.args.get('model_version')}"
            " produced successfully."
        )
    app_context.logger.info(f"Output path: {output_path}")


if __name__ == "__main__":
    main()
