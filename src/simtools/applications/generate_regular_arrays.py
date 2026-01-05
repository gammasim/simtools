#!/usr/bin/python3

"""
Make a regular array of telescopes and save it as astropy table.

The arrays consist of one telescope at the center of the array and or of 4 telescopes
in a square grid. These arrays are used for trigger rate simulations.

The array layout files created will be available at the data/layout directory.

Command line arguments
----------------------
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
from simtools.layout.array_layout_utils import create_regular_array

# Telescope distances for 4 tel square arrays
# !HARDCODED
telescope_distance = {"LST": 57.5 * u.m, "MST": 70 * u.m, "SST": 80 * u.m}


def _parse():
    config = configurator.Configurator(
        label=get_application_label(__file__),
        description=(
            "Generate a regular array of telescope and save as astropy table.\n"
            "Default telescope distances for 4 telescope square arrays are: \n"
            f"  LST: {telescope_distance['LST']}\n"
            f"  MST: {telescope_distance['MST']}\n"
            f"  SST: {telescope_distance['SST']}\n"
        ),
    )
    return config.initialize(
        db_config=False, simulation_model=["site", "model_version"], output=True
    )


def main():
    """Create layout array files (ecsv) of regular arrays."""
    app_context = startup_application(_parse)

    if app_context.args["site"] == "South":
        array_list = ["1SST", "4SST", "1MST", "4MST", "1LST", "4LST"]
    else:
        array_list = ["1MST", "4MST", "1LST", "4LST"]

    for array_name in array_list:
        app_context.logger.info(f"Processing array {array_name}")

        array_table = create_regular_array(
            array_name, app_context.args["site"], telescope_distance=telescope_distance
        )

        output_file = app_context.args.get("output_file")
        if output_file:
            output_path = Path(output_file)
            output_file = output_path.with_name(
                f"{output_path.stem}-{app_context.args['site']}-{array_name}{output_path.suffix}"
            )
        writer.ModelDataWriter.dump(
            args_dict=app_context.args,
            output_file=output_file,
            metadata=None,
            product_data=array_table,
        )


if __name__ == "__main__":
    main()
