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
        List of array element positions (ecsv format)
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

    .. code-block:: console

        python applications/print_array_elements.py \
            --array_element_list NorthArray-utm.ecsv \
            --compact corsika

"""

import logging

import simtools.configuration as configurator
import simtools.util.general as gen
from simtools.layout import layout_array


def _parse(description=None):
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

    config = configurator.Configurator(description=description)

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
    return config.initialize(add_workflow_config=False)


def main():

    args_dict = _parse(description=("Print a list of array element positions"))

    _logger = logging.getLogger()
    _logger.setLevel(gen.getLogLevelFromUser(args_dict["log_level"]))

    layout = layout_array.LayoutArray(args_dict=args_dict)
    layout.readTelescopeListFile(args_dict["array_element_list"])
    layout.convertCoordinates()
    if args_dict["export"] is not None:
        layout.exportTelescopeList(
            args_dict["export"], corsikaZ=args_dict["use_corsika_telescope_height"]
        )
    else:
        layout.printTelescopeList(
            compact_printing=args_dict["compact"],
            corsikaZ=args_dict["use_corsika_telescope_height"],
        )


if __name__ == "__main__":
    main()
