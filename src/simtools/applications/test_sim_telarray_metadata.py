#!/usr/bin/python3

"""
Testing metadata reader / comparison.

Temporary application for development use. Will be removed.

.. code-block:: console

    simtools-db-get-array-layouts-from-db --site North --model_version "6.0.0"
      --array_element_list LSTN-01 LSTN-02 MSTN
      --coordinate_system utm
      --output_file telescope_positions-test_layout.ecsv
"""

import logging
from pathlib import Path

import simtools.testing.sim_telarray_metadata as metadata
import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.model.array_model import ArrayModel


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

    return config.initialize(
        db_config=True, simulation_model=["site", "layout", "model_version"], output=True
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

    array_model = ArrayModel(
        mongo_db_config=db_config,
        model_version="6.0.0",
        site="North",
        layout_name="alpha",
        array_elements=None,
    )

    metadata.assert_sim_telarray_metadata(
        "simtools-grid-output/"
        "run000010_gamma_za20deg_azm000deg_North_test_layout_test-production-North",
        array_model,
    )


if __name__ == "__main__":
    main()
