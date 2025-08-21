#!/usr/bin/python3

r"""
Simulate flasher devices using the light emission package.

Run the application in the command line.

Example Usage
-------------

1. Simulate flashers for a telescope:

    .. code-block:: console

        simtools-simulate-flasher --telescope MSTN-04 --site North \
        --flasher FLSN-01 --model_version 6.0.0

Command Line Arguments
----------------------
telescope (str, required)
    Telescope model name (e.g. LSTN-01, MSTN-04, SSTS-04, ...)
site (str, required)
    Site name (North or South).
flasher (str, required)
    Flasher device in array, e.g., FLSN-01.
model_version (str, optional)
    Version of the simulation model.

Example
-------

Simulate flashers for MSTN-04:

.. code-block:: console

    simtools-simulate-flasher --telescope MSTN-04 --site North \
    --flasher FLSN-01 --model_version 6.0.0

Expected Output:
    The simulation will run the light emission package for the flasher
    devices and produce the output files.
"""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.model.flasher_model import FlasherModel
from simtools.model.model_utils import initialize_simulation_models
from simtools.simtel.simulator_light_emission import SimulatorLightEmission


def _parse(label):
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=label,
        description=("Simulate flasher devices using the light emission package."),
    )
    config.parser.add_argument(
        "--flasher",
        help="Flasher device in array associated with a specific telescope, i.e. FLSN-01",
        type=str,
        required=True,
    )
    config.parser.add_argument(
        "--events",
        help="Number of events to simulate (default: 1)",
        type=int,
        default=1,
        required=False,
    )
    return config.initialize(
        db_config=True,
        simulation_model=["telescope", "model_version"],
        require_command_line=True,
    )


def flasher_configs():
    """Return default setup for flasher runs (no distances)."""
    return {"light_source_setup": "layout"}


def main():
    """Run the application."""
    label = Path(__file__).stem
    logger = logging.getLogger(__name__)

    args_dict, db_config = _parse(label)
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    telescope_model, site_model = initialize_simulation_models(
        label=label,
        db_config=db_config,
        site=args_dict["site"],
        telescope_name=args_dict["telescope"],
        model_version=args_dict["model_version"],
    )

    flasher_model = FlasherModel(
        site=args_dict["site"],
        flasher_device_model_name=args_dict["flasher"],
        mongo_db_config=db_config,
        model_version=args_dict["model_version"],
        label=label,
    )

    flasher_cfg = flasher_configs()

    sim_runner = SimulatorLightEmission(
        telescope_model=telescope_model,
        flasher_model=flasher_model,
        site_model=site_model,
        light_emission_config=args_dict,
        light_source_setup=flasher_cfg["light_source_setup"],
        simtel_path=args_dict["simtel_path"],
        light_source_type="flasher",
        label=args_dict["label"],
        test=args_dict.get("test", False),
    )

    sim_runner.run_simulation()

    logger.info("Flasher simulation completed. Use simtools-plot-simtel-events for plots.")


if __name__ == "__main__":
    main()
