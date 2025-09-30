#!/usr/bin/python3
"""
Generate sim_telarray configuration files for a given array.

The applications generates the sim_telarray configuration files for a given array, site,
and model_version using the model parameters stored in the database.

Command line arguments
----------------------
site : str
    Site name (e.g., North, South).
array_layout_name : str
    Name of the layout array (e.g., test_layout, alpha, 4mst, etc.).
array_element_list : list
    List of array elements (e.g., telescopes) to plot (e.g., ``LSTN-01 LSTN-02 MSTN``).

Example
-------
North - 5.0.0:

.. code-block:: console

    simtools-generate-array-config --site North --array_layout_name alpha --model_version 5.0.0

The output is saved in simtools-output/test/model.
"""

from simtools.application_control import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.model.array_model import ArrayModel


def _parse():
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=get_application_label(__file__),
        description="Generate sim_telarray configuration files for a given array.",
    )
    return config.initialize(db_config=True, simulation_model=["site", "layout", "model_version"])


def main():
    """Generate sim_telarray configuration files for a given array."""
    args_dict, db_config, _, _ = startup_application(_parse)

    array_model = ArrayModel(
        label=args_dict["label"],
        model_version=args_dict["model_version"],
        mongo_db_config=db_config,
        site=args_dict.get("site"),
        layout_name=args_dict.get("array_layout_name"),
        array_elements=args_dict.get("array_elements"),
    )
    array_model.print_telescope_list()
    array_model.export_all_simtel_config_files()


if __name__ == "__main__":
    main()
