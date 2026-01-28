#!/usr/bin/python3

"""
Make a regular array of telescopes and save it to file.

Arrays can consist of single (central) telescopes, square grids or star-like patterns.
All telescopes in the array are of the same type and are placed at regular distances.

Output files are saved as astropy tables in ASCII ECSV format and in the simtools format
required to be used for the overwrite model parameter configuration.

Command line arguments
----------------------
telescope_type (str)
    Type of telescope (e.g., LST, MST, SST).
n_telescopes (int)
    Number of telescopes in the array.
telescope_distance (float)
    Distance between telescopes in the array (in meters).
array_shape (str)
    Shape of the array ('square', 'star').
site (str, required)
    observatory site (e.g., North or South).
model_version (str, optional)
    Model version to use (e.g., 6.0.0). If not provided, the latest version is used.

Example
-------
Runtime < 10 s.

.. code-block:: console

    simtools-generate-regular-arrays --site=North
"""

from pathlib import Path

import astropy.units as u

import simtools.data_model.model_data_writer as writer
from simtools.application_control import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.layout.array_layout_utils import create_regular_array, write_array_elements_info_yaml


def _parse():
    config = configurator.Configurator(
        label=get_application_label(__file__),
        description=("Generate a regular array of telescope and save as astropy table."),
    )
    config.parser.add_argument(
        "--telescope_type",
        help="Type of telescope (e.g., LST, MST, SST).",
        type=str,
        default="LST",
    )
    config.parser.add_argument(
        "--n_telescopes",
        help="Number of telescopes in the array.",
        type=int,
        default=4,
    )
    config.parser.add_argument(
        "--telescope_distance",
        help="Distance between telescopes in the array (in meters).",
        type=float,
        default=50.0,
    )
    config.parser.add_argument(
        "--array_shape",
        help="Shape of the array (e.g., 'square', 'star').",
        type=str,
        default="square",
        choices=["square", "star"],
    )
    return config.initialize(
        db_config=False, simulation_model=["site", "model_version"], output=True
    )


def main():
    """Create layout array files of regular arrays."""
    app_context = startup_application(_parse)

    n_tel = app_context.args["n_telescopes"]
    tel_type = app_context.args["telescope_type"]
    tel_dist = app_context.args["telescope_distance"] * u.m
    shape = app_context.args["array_shape"]

    array_name = f"{n_tel}_{tel_type}_{shape}"
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

        data_writer = writer.ModelDataWriter(
            product_data_file=output_file,
            product_data_format=app_context.args.get("output_file_format", "ascii.ecsv"),
            args_dict=app_context.args,
        )
        data_writer.write(metadata=None, product_data=array_table)

        write_array_elements_info_yaml(
            array_table,
            app_context.args["site"],
            app_context.args["model_version"],
            Path(data_writer.product_data_file).with_suffix(".info.yml"),
        )


if __name__ == "__main__":
    main()
