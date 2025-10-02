#!/usr/bin/python3
"""Convert and print a list of array element positions in different coordinate systems.

Description
-----------

Convert array element positions in different CTAO coordinate systems.
Available coordinate systems are:

1. UTM system
2. ground system (similar to sim_telarray system with x-axis pointing toward geographic north
   and y-axis pointing towards the west); altitude relative to the CORSIKA observation level.
   Altitude is the height of the elevation rotation axis (plus some possible mirror offset).
3. Mercator system

Command line arguments
----------------------
input (str)
    File name with list of array element positions.
    Input can be given as astropy table file (ecsv) or a single array element in
    a json file.
print (str)
    Print in requested coordinate system; possible are ground, utm, mercator
export (str)
    Export array element list to file in requested coordinate system;
      possible are ground, utm, mercator
select_assets (str)
    Select a subset of array elements / telescopes (e.g., MSTN, LSTN)

Example
-------
Convert a list of array elements using a list of telescope positions in UTM coordinates.

.. code-block:: console

    simtools-convert-geo-coordinates-of-array-elements
        --input tests/resources/telescope_positions-North-utm.ecsv
        --print ground

The converted list of telescope positions in ground coordinates is printed to the screen.

The following example converts a list of telescope positions in UTM coordinates
and writes the output to a file in ground (sim_telarray) coordinates. Also selects
only a subset of the array elements (telescopes; ignore calibration devices):

.. code-block:: console

    simtools-convert-geo-coordinates-of-array-elements
        --input tests/resources/telescope_positions-North-utm.ecsv
        --export ground
        --select_assets LSTN

Expected output is a ecsv file in the directory printed to the screen.

"""

import simtools.data_model.model_data_writer as writer
from simtools.application_control import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.layout import array_layout


def _parse():
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=get_application_label(__file__),
        description="Print a list of array element positions",
    )

    config.parser.add_argument(
        "--input",
        help="list of array element positions",
        required=True,
    )
    config.parser.add_argument(
        "--input_meta",
        help="meta data file associated to input data",
        type=str,
        required=False,
    )
    config.parser.add_argument(
        "--print",
        help="print list of positions in requested coordinate system",
        required=False,
        default="",
        choices=[
            "ground",
            "utm",
            "mercator",
        ],
    )
    config.parser.add_argument(
        "--export",
        help="export array element list to file (in requested coordinate system)",
        required=False,
        default=None,
        choices=[
            "ground",
            "utm",
            "mercator",
        ],
    )
    config.parser.add_argument(
        "--select_assets",
        help="select a subset of assets (e.g., MSTN, LSTN)",
        required=False,
        default=None,
        nargs="+",
    )
    config.parser.add_argument(
        "--skip_input_validation",
        help="skip input data validation against schema",
        default=False,
        required=False,
        action="store_true",
    )
    return config.initialize(
        output=True,
        require_command_line=True,
        db_config=True,
        simulation_model=["model_version", "parameter_version", "site"],
    )


def main():
    """Print a list of array elements."""
    app_context = startup_application(_parse)

    if app_context.args.get("input", "").endswith(".json"):
        site = app_context.args.get("site", None)
        metadata, validate_schema_file = None, None
    else:
        metadata = MetadataCollector(
            args_dict=app_context.args, model_parameter_name="array_coordinates"
        )
        site = metadata.get_site(from_input_meta=True)
        validate_schema_file = metadata.get_data_model_schema_file_name()

    layout = array_layout.ArrayLayout(
        mongo_db_config=app_context.db_config,
        model_version=app_context.args["model_version"],
        site=site,
        telescope_list_file=app_context.args["input"],
        telescope_list_metadata_file=app_context.args["input_meta"],
        validate=not app_context.args["skip_input_validation"],
    )
    layout.select_assets(app_context.args["select_assets"])
    layout.convert_coordinates()

    if app_context.args["export"] is not None:
        product_data = (
            layout.export_one_telescope_as_json(
                crs_name=app_context.args["export"],
                parameter_version=app_context.args.get("parameter_version"),
            )
            if app_context.args.get("input", "").endswith(".json")
            else layout.export_telescope_list_table(crs_name=app_context.args["export"])
        )
        writer.ModelDataWriter.dump(
            args_dict=app_context.args,
            metadata=metadata,
            product_data=product_data,
            validate_schema_file=validate_schema_file,
        )
    else:
        layout.print_telescope_list(
            crs_name=app_context.args["print"],
        )


if __name__ == "__main__":
    main()
