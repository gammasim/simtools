#!/usr/bin/python3

r"""
Simulate flasher devices using native LightEmission bindings.

This application demonstrates the native C++ backend for light emission simulation,
which directly calls the sim_telarray LightEmission library instead of using subprocess.
Falls back gracefully to subprocess if native bindings are not available.

The native backend eliminates subprocess overhead and provides direct access to
LightEmission functionality like ff-1m and (future) xyzls.

Example Usage
-------------

1. Simulate flashers using native backend (if available):

    .. code-block:: console

        simtools-simulate-flasher-native \
        --light_source MSFx-FlashCam --model_version 6.0 \
        --telescope MSTS-04 --site South --number_of_events 10 \
        --array_layout_name subsystem_msts


2. Force subprocess fallback:

    .. code-block:: console

        simtools-simulate-flasher-native \
        --light_source MSFx-FlashCam --model_version 6.0 \
        --telescope MSTS-04 --site South \
        --force_subprocess

Command Line Arguments
----------------------
telescope (str, required)
    Telescope model name (e.g. LSTN-01, MSTN-04, SSTS-04, ...)
site (str, required)
    Site name (North or South).
light_source (str, required)
    Calibration light source, e.g., MSFx-FlashCam
number_of_events (int, optional):
    Number of events to simulate (default: 1).
output_prefix (str, optional):
    Prefix for output files (default: empty).
model_version (str, optional)
    Version of the simulation model.
force_subprocess (bool, optional)
    Force use of subprocess even if native backend is available.
"""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.light_emission.native_backend import HAS_NATIVE
from simtools.simtel.simulator_light_emission import SimulatorLightEmission


def _parse(label):
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=label, description="Simulate flasher devices using native LightEmission bindings."
    )
    config.parser.add_argument(
        "--light_source",
        help="Flasher device associated with a specific telescope, i.e. MSFx-FlashCam",
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
    config.parser.add_argument(
        "--force_subprocess",
        help="Force subprocess mode even if native backend is available",
        action="store_true",
        default=False,
    )
    return config.initialize(
        db_config=True,
        simulation_model=["site", "layout", "telescope", "model_version"],
        simulation_configuration={
            "sim_telarray_configuration": ["all"],
        },
    )


def main():
    """Simulate flasher devices using native backend."""
    label = Path(__file__).stem

    args_dict, db_config = _parse(label)
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    # Check native backend availability
    use_native = HAS_NATIVE and not args_dict.get("force_subprocess", False)
    backend_type = "native C++" if use_native else "subprocess"

    logger.info(
        f"Flasher simulation for telescope {args_dict['telescope']} "
        f"with light source {args_dict['light_source']} "
        f"({args_dict['number_of_events']} events, backend: {backend_type})"
    )

    if not HAS_NATIVE:
        logger.info("Native LightEmission backend not available, using subprocess fallback")
    elif args_dict.get("force_subprocess"):
        logger.info("Native backend available but subprocess forced by --force_subprocess")

    # Configure light emission for full simulation with native backend preference
    light_emission_config = args_dict.copy()
    light_emission_config.update(
        {
            "light_source_type": "flat_fielding",  # ff-1m mode
            "use_native_lightemission": use_native,
        }
    )

    # Create simulator
    simulator = SimulatorLightEmission(
        light_emission_config=light_emission_config,
        db_config=db_config,
        label=args_dict.get("label"),
    )

    # Run simulation
    output_file = simulator.simulate()

    logger.info("Flasher simulation completed successfully")
    logger.info(f"Output file: {output_file}")

    # Show performance info if using native
    if use_native:
        logger.info("✓ Used native C++ LightEmission bindings")
    else:
        logger.info("✓ Used subprocess fallback")


if __name__ == "__main__":
    main()
