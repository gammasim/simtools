#!/usr/bin/python3
"""
    Read array element positions from file and write model parameter files for each element.

    This is an application for experts and should not be used by the general user.
    Reading of input is fine-tuned to the array element files as provided by CTAO.

    Writes one model parameter file per array element into a directory structure compatible
    with the simtools model parameter repository.

    Command line arguments

    input : str
        File containing a table of array element positions.
    repository_path : str
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
            --input tests/resources/telescope_positions-North-ground.ecsv \
            --output_path /path/to/repository \
            --parameter_version 0.1.0 \
            --coordinate_system ground

    Add array element positions to repository (utm coordinates):

    .. code-block:: console

        simtools-maintain-simulation-model-write-array-element-positions \
            --input tests/resources/telescope_positions-North-utm.ecsv \
            --output_path /path/to/repository \
            --parameter_version 0.1.0 \
            --coordinate_system utm

"""

from simtools.application_control import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.layout.array_layout_utils import write_array_elements_from_file_to_repository


def _parse():
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=get_application_label(__file__),
        description="Add array element positions to model parameter repository",
    )
    config.parser.add_argument(
        "--input",
        help="File containing a table of array element positions.",
        required=False,
    )
    config.parser.add_argument(
        "--coordinate_system",
        help="Coordinate system of array element positions (utm or ground).",
        default="ground",
        required=False,
        type=str,
        choices=["ground", "utm"],
    )

    return config.initialize(db_config=True, output=True, simulation_model=["parameter_version"])


def main():
    """Application main."""
    app_context = startup_application(_parse)

    write_array_elements_from_file_to_repository(
        coordinate_system=app_context.args["coordinate_system"],
        input_file=app_context.args["input"],
        repository_path=app_context.args["output_path"],
        parameter_version=app_context.args["parameter_version"],
    )


if __name__ == "__main__":
    main()
