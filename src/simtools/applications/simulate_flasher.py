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
TODO fix naming
    Flasher device in array, e.g., FLSN-01.
number_of_events (int, optional):
    Number of events to simulate (default: 1).
output_prefix (str, optional):
    Prefix for output files (default: empty).
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
        # TODO fix naming
        help="Flasher device in array associated with a specific telescope, i.e. FLSN-01",
        type=str,
        required=True,
    )
    config.parser.add_argument(
        "--number_events",
        help="Number of number_events to simulate (default: 1)",
        type=int,
        default=1,
        required=False,
    )
    config.parser.add_argument(
        "--output_prefix",
        help="Prefix for output files (default: empty)",
        type=str,
        default=None,
        required=False,
    )

    return config.initialize(
        db_config=True,
        simulation_model=["telescope", "model_version"],
        require_command_line=True,
    )


def main():
    """Run the application."""
    label = Path(__file__).stem
    logger = logging.getLogger(__name__)

    args_dict, db_config = _parse(label)
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    logger.info(
        f"Flasher simulation for telescope {args_dict['telescope']} "
        f" with flasher {args_dict['flasher']}."
    )
    telescope_model, site_model, calibration_model = initialize_simulation_models(
        label=label,
        db_config=db_config,
        site=args_dict["site"],
        telescope_name=args_dict["telescope"],
        calibration_device_name=args_dict["flasher"],
        model_version=args_dict["model_version"],
    )

    sim_runner = SimulatorLightEmission(
        telescope_model=telescope_model,
        calibration_model=calibration_model,
        site_model=site_model,
        light_emission_config=args_dict,
        simtel_path=args_dict["simtel_path"],
        label=args_dict["label"],
        test=args_dict.get("test", False),
    )

    sim_runner.run_simulation()

    logger.info("Flasher simulation completed.")


if __name__ == "__main__":
    main()
