#!/usr/bin/python3

r"""
Simulate pedestal events for calibration purposes.

Example
-------

Simulate pedestal events for MSTN-04 at the North site:

    .. code-block:: console

        simtools-simulate-pedestal-events --telescope MSTN-04 --site North \
        --number_of_events 1000 --model_version 6.0.0

Command Line Arguments
----------------------
telescope (str, required)
    Telescope model name (e.g. LSTN-01, SSTS-design, SSTS-25, ...)
site (str, required)
    Site name (North or South).
model_version (str, optional)
    Version of the simulation model.

"""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.model.model_utils import initialize_simulation_models


def _parse(label):
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=label,
        description=("Simulate pedestal events for calibration purposes."),
    )
    return config.initialize(
        db_config=True,
        simulation_model=["telescope", "model_version"],
    )


def main():  # noqa: D103
    label = Path(__file__).stem
    args_dict, db_config = _parse(label)

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    telescope_model, site_model = initialize_simulation_models(
        label=label,
        db_config=db_config,
        site=args_dict["site"],
        telescope_name=args_dict["telescope"],
        model_version=args_dict["model_version"],
    )



if __name__ == "__main__":
    main()
