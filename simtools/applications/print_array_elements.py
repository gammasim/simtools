#!/usr/bin/python3
"""
    Summary
    -------
    Print a list of array element positions in different CTAO coordinate \
    systems.

    Available coordinate systems are:
    1. UTM system
    2. CORSIKA coordinates
    3. Mercator system

    Command line arguments
    ----------------------
    array_element_list (str)
        File name with list of array element positions (ecsv format)
    compact (str)
        Compact output (in requested coordinate system; possible are corsika,utm,mercator)
    export (str)
        Export array element list to file (in requested coordinate system; \
            possible are corsika,utm,mercator)
    use_corsika_telescope_height (bool)
        Use CORSIKA coordinates for telescope heights (requires CORSIKA observeration level)


    Example
    -------
    Print a list of array elements using a list of telescope positions in UTM coordinates.

    Example:

    Run the application:

    .. code-block:: console

        simtools-print-array-elements \
            --array_element_list tests/resources/telescope_positions-South-4MST.ecsv \
            --compact corsika

    Expected final print-out message:

    .. code-block:: console

    telescope_name pos_x pos_y altitude
    MST-01      -0.02      -0.00    2162.00
    MST-02       1.43     151.02    2163.00
    MST-03      -1.47    -151.02    2169.00
    MST-04     150.72      73.57    2159.00



"""

import logging
from pathlib import Path

import simtools.util.general as gen
from simtools.configuration import configurator
from simtools.layout import layout_array


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
        "--array_element_list",
        help="list of array element positions (ecsv format)",
        required=True,
    )
    config.parser.add_argument(
        "--compact",
        help="compact output (in requested coordinate system)",
        required=False,
        default="",
        choices=[
            "corsika",
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
            "corsika",
            "utm",
            "mercator",
        ],
    )
    config.parser.add_argument(
        "--use_corsika_telescope_height",
        help="Use CORSIKA coordinates for telescope heights (requires CORSIKA observeration level)",
        required=False,
        default=False,
        action="store_true",
    )
    return config.initialize()


def main():
    label = Path(__file__).stem
    args_dict, _ = _parse(label, description="Print a list of array element positions")

    _logger = logging.getLogger()
    _logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    layout = layout_array.LayoutArray(telescope_list_file=args_dict["array_element_list"])
    layout.convert_coordinates()
    if args_dict["export"] is not None:
        layout.export_telescope_list(
            crs_name=args_dict["export"],
            corsika_z=args_dict["use_corsika_telescope_height"],
        )
    else:
        layout.print_telescope_list(
            compact_printing=args_dict["compact"],
            corsika_z=args_dict["use_corsika_telescope_height"],
        )


if __name__ == "__main__":
    main()
