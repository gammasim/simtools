#!/usr/bin/python3
"""
    Summary
    -------
    Print a list of array element positions in different CTAO coordinate \
    systems.

    Coordinate systems are:
    1. UTM system
    2. CORSIKA coordinates
    3. Mercator system

    Command line arguments
    ----------------------
    Example
    -------
    Printing a list of array elements using a list of telescope positions \
    in UTM coordinates.

    Example:

    .. code-block:: console

        python applications/print_array_elements.py \
            --array_elements NorthArray-utm.ecsv \
            --compact corsika

"""

import logging

import simtools.layout.layout_array as layout_array
import simtools.util.commandline_parser as argparser
import simtools.util.general as gen


def parse(description=None):
    """
    Parse command line configuration

    Parameters
    ----------
    description: str
        description of application.

    Returns
    -------
    CommandLineParser
        command line parser object

    """

    parser = argparser.CommandLineParser(description=description)

    parser.add_argument(
        "--array_element_list", help="list of array element positions (ecsv format)"
    )
    parser.add_argument(
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
    parser.add_argument(
        "--export",
        help="export array element list to file",
        required=False,
        default=None,
        choices=[
            "corsika",
            "utm",
            "mercator",
        ],
    )
    parser.add_argument(
        "--use_corsika_telescope_height",
        help="Use CORSIKA coordinates for telescope height (requires CORSIKA observeration level)",
        required=False,
        default=False,
        action="store_true",
    )
    parser.initialize_default_arguments(add_workflow_config=False)
    return parser.parse_args()


def main():

    args = parse(description=("Print a list of array element positions"))

    _logger = logging.getLogger()
    _logger.setLevel(gen.getLogLevelFromUser(args.logLevel))

    layout = layout_array.LayoutArray()
    layout.readTelescopeListFile(args.array_element_list)

    layout.convertCoordinates()
    layout.printTelescopeList(
        compact_printing=args.compact, corsikaZ=args.use_corsika_telescope_height
    )
    if args.export is not None:
        layout.exportTelescopeList(args.export)


if __name__ == "__main__":
    main()
