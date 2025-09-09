#!/usr/bin/python3

r"""
Simulate illuminator (distant calibration light source).

Illuminators are calibration light sources not attached to a particular telescope.
Two types of illuminators are supported:

1. Illuminator as foreseen at CTAO with fixed positions as defined in the simulation models
   database.
2. Illuminator at a configurable position relative to the array center. Note that in this case
   the telescope pointing is fixed towards zenith.

Example Usage
-------------

1. Simulate illuminator with positions as defined in the simulation models database:

    .. code-block:: console

        simtools-simulate-illuminator --light_source ILLN-01 \
        --telescope MSTN-04 --site North \
        --model_version 6.0.0

2. Simulate at a configurable position (1km above array center) and pointing downwards:

    .. code-block:: console

        simtools-simulate-illuminator --light_source ILLN-01 \
        --light_source_position 0. 0. 1000. \
        --light_source_pointing 0. 0. -1. \
        --telescope MSTN-04 --site North \
        --model_version 6.0.0

Command Line Arguments
----------------------
light_source (str, optional)
    Illuminator in array, e.g., ILLN-01.
number_of_events (int, optional)
    Number of events to simulate.
telescope (str, required)
    Telescope model name (e.g. LSTN-01, SSTS-design, SSTS-25, ...)
site (str, required)
    Site name (North or South).
model_version (str, optional)
    Version of the simulation model.
light_source_position (float, float, float, optional)
    Light source position (x,y,z) relative to the array center (ground coordinates) in
    m. If not set, the position from the simulation model is used.
light_source_pointing (float, float, float, optional)
    Light source pointing direction. If not set, the pointing from the simulation model is used.
output_prefix (str, optional)
    Prefix for output files (default: empty).
"""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.simtel.simulator_light_emission import SimulatorLightEmission


def _parse(label):
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=label,
        description=(
            "Simulate light emission by a calibration light source (not attached to a telescope)."
        ),
    )
    config.parser.add_argument(
        "--light_source",
        help="Illuminator name, i.e. ILLN-design",
        type=str,
        default=None,
        required=True,
    )
    configurable_light_source_args = config.parser.add_argument_group(
        "Configurable light source position and pointing (override simulation model values)"
    )
    configurable_light_source_args.add_argument(
        "--light_source_position",
        help="Light source position (x,y,z) relative to the array center (ground coordinates) in m",
        metavar=("X", "Y", "Z"),
        nargs=3,
        required=False,
    )
    configurable_light_source_args.add_argument(
        "--light_source_pointing",
        help=(
            "Light source pointing direction "
            "(Example for pointing downwards: --light_source_pointing 0 0 -1)"
        ),
        metavar=("X", "Y", "Z"),
        nargs=3,
        required=False,
    )
    config.parser.add_argument(
        "--number_of_events",
        help="Number of events to simulate",
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
    """Simulate light emission from illuminator."""
    label = Path(__file__).stem

    args_dict, db_config = _parse(label)
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    light_source = SimulatorLightEmission(
        light_emission_config=args_dict,
        db_config=db_config,
        label=label,
    )
    light_source.simulate()


if __name__ == "__main__":
    main()
