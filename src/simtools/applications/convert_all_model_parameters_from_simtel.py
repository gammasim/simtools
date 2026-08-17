#!/usr/bin/python3
"""Convert all simulation model parameters from sim_telarray format to simtools json format."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.model import sim_telarray_parameter_converter

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "simtel_cfg_file", help="File name for sim_telarray configuration", type=str, required=True
    ),
    cli.ArgumentDefinition(
        "simtel_telescope_name",
        help="Name of the telescope in the sim_telarray configuration file",
        type=str,
        required=True,
    ),
    cli.ArgumentDefinition(
        "skip_parameter", help="List of parameters to be skipped.", type=str, nargs="*", default=[]
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
    ),
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    sim_telarray_parameter_converter.run_conversion_workflow(app_context)


if __name__ == "__main__":
    main()
