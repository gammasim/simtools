#!/usr/bin/python3
"""Produce markdown reports for the simulation model.

Report types: array_element (per telescope), calibration (calibration devices),
model_parameter (model parameters per telescope) and simulation_configuration
(simulation software configuration parameters).
"""

from pathlib import Path

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.reporting.docs_auto_report_generator import ReportGenerator
from simtools.reporting.docs_read_parameters import ReadParameters

_REPORT_TYPES = ("array_element", "calibration", "model_parameter", "simulation_configuration")

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "report",
        help="Type of report to produce.",
        type=str,
        choices=list(_REPORT_TYPES),
        required=True,
    ),
    cli.ALL_MODEL_VERSIONS,
    cli.ArgumentDefinition(
        "all_telescopes", action="store_true", help="Produce reports for all telescopes."
    ),
    cli.ArgumentDefinition("all_sites", action="store_true", help="Produce reports for all sites."),
    cli.ArgumentDefinition(
        "observatory",
        action="store_true",
        help="Produce array_element reports for an observatory at a given site.",
    ),
    cli.SIMULATION_SOFTWARE,
    cli.MODEL_VERSION,
    cli.OVERWRITE_MODEL_PARAMETERS,
    cli.SITE,
    cli.TELESCOPE,
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    model_repository=True,
    arguments=(*_ARGUMENTS, *cli.OUTPUT_PATH_ARGUMENTS),
    usage="simtools-docs-produce-reports --report <report_type> [options]",
    needs=("model", "corsika"),
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()
    output_path = app_context.paths.get_output_directory()

    producers = {
        "array_element": _produce_array_element_reports,
        "calibration": _produce_calibration_reports,
        "model_parameter": _produce_model_parameter_reports,
        "simulation_configuration": _produce_simulation_configuration_reports,
    }
    producers[app_context.args["report"]](app_context, output_path)


def _produce_array_element_reports(app_context, output_path):
    """Produce array element reports (single telescope or all telescopes/sites/versions)."""
    if any(
        [
            app_context.args.get("all_telescopes"),
            app_context.args.get("all_sites"),
            app_context.args.get("all_model_versions"),
        ]
    ):
        ReportGenerator(
            app_context.args, output_path, app_context.model_reader, context=app_context
        ).auto_generate_array_element_reports()

    else:
        model_version = app_context.args["model_version"]
        ReadParameters(
            app_context.args,
            Path(output_path / f"{model_version}"),
            app_context.model_reader,
            context=app_context,
        ).produce_array_element_report()

        app_context.logger.info(
            f"Markdown report generated for {app_context.args['site']}"
            f" Telescope {app_context.args['telescope']} (v{model_version}):"
            f" {output_path}"
        )


def _produce_calibration_reports(app_context, output_path):
    """Produce calibration-device reports."""
    generator = ReportGenerator(
        args=app_context.args,
        output_path=output_path,
        model_reader=app_context.model_reader,
        context=app_context,
    )
    generator.auto_generate_calibration_reports()

    if app_context.args.get("all_model_versions"):
        app_context.logger.info("Calibration reports for all model versions produced successfully.")
    else:
        app_context.logger.info(
            f"Calibration reports for model version {app_context.args.get('model_version')}"
            " produced successfully."
        )
    app_context.logger.info(f"Output path: {output_path}")


def _produce_model_parameter_reports(app_context, output_path):
    """Produce model parameter reports per telescope."""
    if any([app_context.args.get("all_telescopes"), app_context.args.get("all_sites")]):
        ReportGenerator(
            app_context.args,
            output_path,
            app_context.model_reader,
            context=app_context,
        ).auto_generate_parameter_reports()

    else:
        ReadParameters(
            app_context.args,
            output_path,
            app_context.model_reader,
            context=app_context,
        ).produce_model_parameter_reports()

        app_context.logger.info(
            f"Markdown report generated for {app_context.args['site']}"
            f" Telescope {app_context.args['telescope']}: {output_path}"
        )


def _produce_simulation_configuration_reports(app_context, output_path):
    """Produce simulation configuration reports."""
    report_generator = ReportGenerator(
        args=app_context.args,
        output_path=output_path,
        model_reader=app_context.model_reader,
        context=app_context,
    )
    report_generator.auto_generate_simulation_configuration_reports()

    app_context.logger.info(
        f"Configuration reports for {app_context.args.get('simulation_software')} "
        "produced successfully."
    )
    app_context.logger.info(f"Output path: {output_path}")


if __name__ == "__main__":
    main()
