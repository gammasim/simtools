#!/usr/bin/python3

"""Make a regular array of telescopes and save it to file."""

from pathlib import Path

import astropy.units as u

import simtools.data_model.model_data_writer as writer
from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.layout.array_layout_utils import create_regular_array, write_array_elements_info_yaml

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "telescope_type", help="Type of telescope (e.g., LST, MST, SST).", type=str, default="LST"
    ),
    cli.ArgumentDefinition(
        "number_of_telescopes", help="Number of telescopes in the array.", type=int, default=4
    ),
    cli.ArgumentDefinition(
        "telescope_distance",
        help="Distance between telescopes in the array (in meters).",
        type=float,
        default=50.0,
    ),
    cli.ArgumentDefinition(
        "array_shape",
        help="Shape of the array (e.g., 'square', 'star').",
        type=str,
        default="square",
        choices=["square", "star"],
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        cli.MODEL_VERSION,
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.SITE,
        *cli.OUTPUT_PATH_ARGUMENTS,
        *cli.OUTPUT_ARGUMENTS,
    ),
    initialize_output=True,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    n_tel = app_context.args["number_of_telescopes"]
    tel_type = app_context.args["telescope_type"]
    tel_dist = app_context.args["telescope_distance"] * u.m
    shape = app_context.args["array_shape"]

    array_name = f"{n_tel}{tel_type}-{shape}"
    app_context.logger.info(f"Processing array {array_name}")

    array_table = create_regular_array(
        array_name,
        app_context.args["site"],
        n_telescopes=n_tel,
        telescope_type=tel_type,
        telescope_distance=tel_dist,
        shape=shape,
    )

    output_file = app_context.args.get("output_file")
    if output_file:
        output_path = Path(output_file)
        output_file = output_path.with_name(
            f"{output_path.stem}-{app_context.args['site']}-{array_name}{output_path.suffix}"
        )

        writer.ModelDataWriter.write_product_data(
            output_file=output_file,
            output_file_format=app_context.args.get("output_file_format", "ascii.ecsv"),
            product_data=array_table,
        )

        write_array_elements_info_yaml(
            array_table,
            app_context.args["site"],
            app_context.args["model_version"],
            app_context.io_handler.get_output_directory()
            / Path(output_file).with_suffix(".info.yml"),
        )


if __name__ == "__main__":
    main()
