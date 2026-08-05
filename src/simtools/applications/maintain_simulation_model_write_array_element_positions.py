#!/usr/bin/python3
"""
    Read array element positions from file and write model parameter files for each element.

    This is an application for experts and should not be used by the general user.
    Reading of input is fine-tuned to the array element files as provided by CTAO.

    Writes one model parameter file per array element into a directory structure compatible
    with the simtools model parameter repository.

    Command line arguments

    array_element_positions_file : str
        File containing a table of array element positions.
    simulation_models_path : Path
        Path of local copy of model parameter repository.
    parameter_version : str
        Parameter version.
    coordinate_system : str
        Coordinate system of array element positions (ground or utm).

    Examples
    --------
    Add array element positions to repository (ground coordinates):

    .. code-block:: console

        simtools-maintain-simulation-model-write-array-element-positions \
            --array_element_positions_file tests/resources/telescope_positions-North-ground.ecsv \
            --simulation_models_path /path/to/repository \
            --parameter_version 0.1.0 \
            --coordinate_system ground

    Add array element positions to repository (utm coordinates):

    .. code-block:: console

        simtools-maintain-simulation-model-write-array-element-positions \
            --array_element_positions_file tests/resources/telescope_positions-North-utm.ecsv \
            --simulation_models_path /path/to/repository \
            --parameter_version 0.1.0 \
            --coordinate_system utm

"""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.layout.array_layout_utils import write_array_elements_from_file_to_repository

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "array_element_positions_file",
        help="File containing a table of array element positions.",
        required=True,
    ),
    cli.SIMULATION_MODELS_PATH(required=True),
    cli.ArgumentDefinition(
        "coordinate_system",
        help="Coordinate system of array element positions (utm or ground).",
        default="ground",
        required=False,
        type=str,
        choices=["ground", "utm"],
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        cli.PARAMETER_VERSION,
        cli.OVERWRITE_MODEL_PARAMETERS,
    ),
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    write_array_elements_from_file_to_repository(
        coordinate_system=app_context.args["coordinate_system"],
        input_file=app_context.args["array_element_positions_file"],
        repository_path=app_context.args["simulation_models_path"],
        parameter_version=app_context.args["parameter_version"],
    )


if __name__ == "__main__":
    main()
