#!/usr/bin/python3
"""
    Summary
    -------
    Convert and print a list of array element positions in different coordinate \
    systems relevant for CTAO.

    Description
    -----------

    This application converts a list of array element positions in different CTAO \
    coordinate systems.

    Available coordinate systems are:

    1. UTM system
    2. ground system (similar to sim_telarray system with x-axis pointing toward \
        geographic north and y-axis pointing towards the west)
    3. Mercator system

    Command line arguments
    ----------------------
    input (str)
        File name with list of array element positions
    compact (str)
        Compact output in requested coordinate system; possible are corsika,utm,mercator
    export (str)
        Export array element list to file in requested coordinate system; \
            possible are ground, utm, mercator
    use_corsika_telescope_height (bool)
        Use CORSIKA coordinates for telescope heights (requires CORSIKA observation level)
    select_assets (str)
        Select a subset of array elements / telescopes (e.g., MSTN, LSTN)


    Example
    -------
    Print a list of array elements using a list of telescope positions in UTM coordinates.

    .. code-block:: console

        simtools-print-array-elements \\
            --input tests/resources/telescope_positions-South-4MST.ecsv \\
            --compact ground

    Expected final print-out message:

    .. code-block:: console

        telescope_name pos_x pos_y altitude
        MST-01      -0.02      -0.00    2162.00
        MST-02       1.43     151.02    2163.00
        MST-03      -1.47    -151.02    2169.00
        MST-04     150.72      73.57    2159.00

    The following example converts a list of telescope positions in UTM coordinates \
    and writes the output to a file in ground (sim_telarray) coordinates. Also selects \
    only a subset of the array elements (telescopes; ignore calibration devices):

    .. code-block:: console

        simtools-print-array-elements \\
            --input tests/resources/telescope_positions-North-utm.ecsv \\
            --export ground --use_corsika_telescope_height \\
            --select_assets LSTN, MSTN, SSTN

    Expected output is a ecsv file in the directory printed to the screen.

"""

import logging
from pathlib import Path

import simtools.data_model.model_data_writer as writer
import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.layout import array_layout


def _parse(label=None, description=None):
    """
    Parse command line configuration

    Parameters
    ----------
    label: str
        Label describing application.
    description: str
        Description of application.

    Returns
    -------
    CommandLineParser
        Command line parser object

    """

    config = configurator.Configurator(label=label, description=description)

    config.parser.add_argument(
        "--input",
        help="list of array element positions",
        required=False,
    )
    config.parser.add_argument(
        "--input_meta",
        help="meta data file associated to input data",
        type=str,
        required=False,
    )
    config.parser.add_argument(
        "--compact",
        help="compact output (in requested coordinate system)",
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
        "--use_corsika_telescope_height",
        help="Use CORSIKA coordinates for telescope heights (requires CORSIKA observation level)",
        required=False,
        default=False,
        action="store_true",
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
    return config.initialize(output=True, require_command_line=True)


def main():
    label = Path(__file__).stem
    data_model_name = "array_coordinates"
    args_dict, _ = _parse(
        label,
        description=f"Print a list of array element positions ({data_model_name})",
    )

    _logger = logging.getLogger()
    _logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    layout = array_layout.ArrayLayout(
        telescope_list_file=args_dict["input"],
        telescope_list_metadata_file=args_dict["input_meta"],
        validate=not args_dict["skip_input_validation"],
    )
    layout.select_assets(args_dict["select_assets"])
    layout.convert_coordinates()

    if args_dict["export"] is not None:
        _metadata = MetadataCollector(args_dict=args_dict, data_model_name=data_model_name)
        writer.ModelDataWriter.dump(
            args_dict=args_dict,
            metadata=_metadata.top_level_meta,
            product_data=layout.export_telescope_list_table(
                crs_name=args_dict["export"],
                corsika_z=args_dict["use_corsika_telescope_height"],
            ),
            validate_schema_file=_metadata.get_data_model_schema_file_name(),
        )
    else:
        layout.print_telescope_list(
            compact_printing=args_dict["compact"],
            corsika_z=args_dict["use_corsika_telescope_height"],
        )


if __name__ == "__main__":
    main()
