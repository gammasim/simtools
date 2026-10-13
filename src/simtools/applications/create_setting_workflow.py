#!/usr/bin/python3
"""Prepare a simple parameter-setting workflow and optionally run it."""

from pathlib import Path

import yaml

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import argument_helpers
from simtools.configuration import arguments as cli
from simtools.data_model.setting_workflow import create_setting_workflow, run_setting_workflow

APPLICATION = ApplicationDefinition.for_module(
    __name__,
    model_repository=True,
    resolve_sim_software_executables=False,
    setup_io_handler=False,
    excluded_standard_arguments=("runtime_environment_file",),
    arguments=(
        cli.ArgumentDefinition(
            "instrument",
            type=argument_helpers.instrument,
            required=True,
            help="Instrument whose parameter is being set.",
        ),
        cli.ArgumentDefinition("parameter", required=True, help="Model parameter name."),
        cli.ArgumentDefinition("parameter_version", required=True, help="New parameter version."),
        cli.ArgumentDefinition("value", required=True, help="Value, optionally with a unit."),
        cli.ArgumentDefinition("description", required=True, help="Scientific reason or source."),
        cli.ArgumentDefinition(
            "site",
            type=argument_helpers.site,
            help="Site; inferred for instruments with an unambiguous site.",
        ),
        cli.ArgumentDefinition("source_url", help="Reference describing the parameter setting."),
        cli.ArgumentDefinition(
            "model_parameter_schema_version", help="Parameter schema version (default: latest)."
        ),
        cli.OUTPUT_PATH(default=Path(), help="Root of the parameter-setting repository."),
        cli.ArgumentDefinition(
            "workflow_runtime_file",
            aliases=("runtime_environment_file",),
            type=Path,
            help="Runtime YAML to embed in the generated workflow.",
        ),
        cli.ArgumentDefinition("run", action="store_true", help="Run the prepared workflow."),
    ),
)


def main():
    """Prepare the workflow and report the files and production reference."""
    context = APPLICATION.start()
    config_file = create_setting_workflow(context.args, context.model_reader)
    context.logger.info(f"Prepared workflow: {config_file}")
    reference = _production_info_reference(context.args, config_file)
    context.logger.info("info.yml entry:\n%s", yaml.safe_dump(reference, sort_keys=False))
    context.logger.info(
        "Add the following to the corresponding production-info file:\n%s",
        _production_info_text(reference),
    )
    if context.args.get("run"):
        output = run_setting_workflow(config_file, context.args)
        context.logger.info(f"Results: {output}")


def _production_info_reference(args, config_file):
    """Return the production-info entry for the prepared setting."""
    return {
        args["instrument"]: {
            args["parameter"]: {
                "version": args["parameter_version"],
                "activity_id": config_file.parent.name,
            }
        }
    }


def _production_info_text(reference):
    """Format a production-info entry with semantic versions quoted."""
    instrument, parameters = next(iter(reference.items()))
    parameter, setting = next(iter(parameters.items()))
    return (
        f"{instrument}:\n"
        f"  {parameter}:\n"
        f'    version: "{setting["version"]}"\n'
        f"    activity_id: {setting['activity_id']}"
    )


if __name__ == "__main__":
    main()
