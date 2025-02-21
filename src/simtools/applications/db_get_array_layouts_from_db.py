#!/usr/bin/python3

"""
Get list of array layouts or list of elements for a given layout as defined in the db.

To get the list of pre-defined array layouts, use ``--list_available_layouts``.

To get the list of array elements for a given layout, use ``--array_layout_name``.

To get the positions for a set of array elements, use ``--array_element_list``.
Listing of array elements follows this logic:

* explicit listing: e.g., ``-array_element_list MSTN-01, MSTN05``
* listing of types: e.g, ``-array_element_list MSTN`` plots all telescopes of type MSTN.

Command line arguments
----------------------
list_available_layouts : bool, optional
    List available layouts in the database.
array_layout_name : str
    Name of the layout array (e.g., test_layout, alpha, 4mst, etc.).
array_element_list : list
    List of array elements (e.g., telescopes) to plot (e.g., ``LSTN-01 LSTN-02 MSTN``).
coordinate_system : str, optional
    Coordinate system for the array layout (ground or utm).
output_file : str, optional
    Name of the output file to be saved as astropy table (ecsv file)

Examples
--------
List pre-defined array layouts.

.. code-block:: console

    simtools-db-get-array-layouts-from-db --site North --model_version "6.0.0"

Retrieve telescope positions for array layout 'test_layout' from database.

.. code-block:: console

    simtools-db-get-array-layouts-from-db --site North --model_version "6.0.0"
        --array_layout_name test_layout

Retrieve telescope positions from database (utm coordinate system) and write to an ecsv files

.. code-block:: console

    simtools-db-get-array-layouts-from-db --site North --model_version "6.0.0"
      --array_element_list LSTN-01 LSTN-02 MSTN
      --coordinate_system utm
      --output_file telescope_positions-test_layout.ecsv
"""

import logging
from pathlib import Path

import simtools.data_model.model_data_writer as writer
import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.model.array_model import ArrayModel
from simtools.model.site_model import SiteModel


def _parse(label, description):
    """
    Parse command line configuration.

    Parameters
    ----------
    label : str
        Label describing the application.
    description : str
        Description of the application.

    Returns
    -------
    CommandLineParser
        Command line parser object.
    """
    config = configurator.Configurator(label=label, description=description)

    input_group = config.parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--list_available_layouts",
        help="List available layouts in the database.",
        action="store_true",
        required=False,
    )
    config.parser.add_argument(
        "--coordinate_system",
        help="Coordinate system for the array layout.",
        type=str,
        required=False,
        default="ground",
        choices=["ground", "utm"],
    )
    return config.initialize(
        db_config=True, simulation_model=["site", "layout", "model_version"], output=True
    )


def _layout_from_db(args_dict, db_config):
    """
    Read array elements and their positions from data base using the layout name.

    Parameters
    ----------
    args_dict : dict
        Dictionary with the command line arguments.
    db_config : dict
        Database configuration.

    Returns
    -------
    astropy.table.Table
        Table with array element positions.
    """
    array_model = ArrayModel(
        mongo_db_config=db_config,
        model_version=args_dict["model_version"],
        site=args_dict["site"],
        layout_name=args_dict.get("array_layout_name", None),
        array_elements=args_dict.get("array_element_list", None),
    )
    return array_model.export_array_elements_as_table(
        coordinate_system=args_dict["coordinate_system"]
    )


def main():
    """Get list of array elements as defined in the db (array layout)."""
    label = Path(__file__).stem
    args_dict, db_config = _parse(
        label,
        "Get list of array elements as defined in the db (array layout).",
    )
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    if args_dict.get("list_available_layouts", False):
        if args_dict.get("site", None) is None:
            raise ValueError("Site must be provided to list available layouts.")
        site_model = SiteModel(
            mongo_db_config=db_config,
            model_version=args_dict["model_version"],
            site=args_dict["site"],
        )
        print(site_model.get_list_of_array_layouts())
    else:
        logger.info("Array layout: %s", args_dict["array_layout_name"])
        layout = _layout_from_db(args_dict, db_config)
        layout.pprint()

        if not args_dict.get("output_file_from_default", False):
            writer.ModelDataWriter.dump(
                args_dict=args_dict,
                output_file=args_dict["output_file"],
                metadata=None,
                product_data=layout,
            )


if __name__ == "__main__":
    main()
