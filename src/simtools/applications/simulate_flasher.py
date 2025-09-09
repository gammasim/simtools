#!/usr/bin/python3

r"""
Simulate flasher devices used e.g. for camera flat fielding.

The flasher simulation allows two different run modes:

1. Direct injection of light into the camera (bypassing the telescope optics).
2. Simulation of the full light path (using the light-emission package from sim_telarray).

The direct injection mode uses a simplified model for the flasher light source. Both run modes
provide events in sim_telarray format that can be processed by standard analysis steps or
visualized using e.g. the 'simtools-plot-simtel-events' application.

Example Usage
-------------

1. Simulate flashers for a telescope (direct injection):

    .. code-block:: console

        simtools-simulate-flasher --run_mode direct_injection \
        --telescope MSTS-04 --site South \
        --flasher MSFx-FlashCam --model_version 6.0.0

2. Simulate flashers for a telescope (detailed simulation):

    .. code-block:: console

        simtools-simulate-flasher --run_mode full_simulation \
        --telescope MSTS-04 --site South \
        --flasher MSFx-FlashCam --model_version 6.0.0

Command Line Arguments
----------------------
run_mode (str, required)
    Run mode, either "direct_injection" or "full_simulation".
telescope (str, required)
    Telescope model name (e.g. LSTN-01, MSTN-04, SSTS-04, ...)
site (str, required)
    Site name (North or South).
flasher (str, required)
    Flasher device in array, e.g., MSFx-FlashCam
number_of_events (int, optional):
    Number of events to simulate (default: 1).
output_prefix (str, optional):
    Prefix for output files (default: empty).
model_version (str, optional)
    Version of the simulation model.
"""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.simtel.simulator_light_emission import SimulatorLightEmission
from simtools.simulator import Simulator


def _parse(label):
    """Parse command line configuration."""
    config = configurator.Configurator(label=label, description="Simulate flasher devices.")
    config.parser.add_argument(
        "--run_mode",
        help="Flasher simulation run mode",
        type=str,
        choices=["direct_injection", "full_simulation"],
        required=True,
        default="direct_injection",
    )
    config.parser.add_argument(
        "--light_source",
        help="Flasher device name associated with a specific telescope, i.e. MSFx-FlashCam",
        type=str,
        required=True,
    )
    config.parser.add_argument(
        "--number_of_events",
        help="Number of flasher events to simulate",
        type=int,
        default=1,
        required=False,
    )
    config.parser.add_argument(
        "--output_prefix",
        help="Prefix for output files",
        type=str,
        default=None,
        required=False,
    )
    return config.initialize(
        db_config=True,
        simulation_model=["site", "layout", "telescope", "model_version"],
        simulation_configuration={
            "corsika_configuration": ["run_number", "azimuth_angle", "zenith_angle"],
            "sim_telarray_configuration": ["all"],
        },
    )


def main():
    """Simulate flasher devices."""
    label = Path(__file__).stem

    args_dict, db_config = _parse(label)
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    logger.info(
        f"Flasher simulation for telescope {args_dict['telescope']} "
        f" with light source {args_dict['light_source']} "
        f" ({args_dict['number_of_events']} events, run mode: {args_dict['run_mode']})"
    )

    if args_dict["run_mode"] == "full_simulation":
        light_source = SimulatorLightEmission(
            light_emission_config=args_dict,
            db_config=db_config,
            label=args_dict.get("label"),
        )
    elif args_dict["run_mode"] == "direct_injection":
        light_source = Simulator(
            args_dict=args_dict,
            db_config=db_config,
            label=args_dict.get("label"),
        )
    else:
        raise ValueError(f"Unsupported run_mode: {args_dict['run_mode']}")

    light_source.simulate()
    logger.info("Flasher simulation completed.")


if __name__ == "__main__":
    main()
