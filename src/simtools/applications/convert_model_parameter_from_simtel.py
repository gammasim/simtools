#!/usr/bin/python3
"""Convert a sim_telarray parameter to the simulation-model format."""

import simtools.data_model.model_data_writer as writer
from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.simtel.simtel_config_reader import SimtelConfigReader

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "schema_file", help="Schema file for model parameter validation", required=True
    ),
    cli.ArgumentDefinition(
        "simtel_cfg_file", help="File name for sim_telarray configuration", type=str, required=True
    ),
    cli.ArgumentDefinition(
        "simtel_telescope_name",
        help="Name of the telescope in the sim_telarray configuration file",
        type=str,
        required=True,
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        cli.PARAMETER_VERSION,
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.SITE,
        cli.TELESCOPE,
        *cli.OUTPUT_PATH_ARGUMENTS,
        *cli.OUTPUT_ARGUMENTS,
    ),
    initialize_output=True,
    setup_io_handler=False,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    simtel_config_reader = SimtelConfigReader(
        schema_file=app_context.args["schema_file"],
        simtel_config_file=app_context.args["simtel_cfg_file"],
        simtel_telescope_name=app_context.args["simtel_telescope_name"],
    )
    app_context.logger.info(f"Simtel parameter: {simtel_config_reader.parameter_dict}")
    if simtel_config_reader.parameter_dict is None or len(simtel_config_reader.parameter_dict) == 0:
        app_context.logger.error("Parameter not found in sim_telarray configuration file.")
        return

    simtel_config_reader.compare_simtel_config_with_schema()

    _json_dict = writer.ModelDataWriter.write_model_parameter(
        parameter_name=simtel_config_reader.parameter_name,
        value=simtel_config_reader.parameter_dict.get(app_context.args["simtel_telescope_name"]),
        instrument=app_context.args["telescope"],
        parameter_version=app_context.args["parameter_version"],
        output_file=app_context.args["output_file"],
        output_path=app_context.args.get("output_path"),
    )
    app_context.logger.info(f"Validated parameter: {_json_dict}")


if __name__ == "__main__":
    main()
