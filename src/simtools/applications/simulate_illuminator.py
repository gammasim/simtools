#!/usr/bin/python3

r"""
Simulate illuminator (distant calibration light source).

Illuminators are calibration light sources not attached to a particular telescope.
Three modes of operation are supported:

1. Single pair: simulate one illuminator-telescope pair with positions from the model database.
2. Single pair with configurable position: override the illuminator position and pointing.
3. Multi-pair: simulate all valid illuminator-telescope pairs from the visibility table
   in parallel, using a configurable number of CPU cores.

Example Usage
-------------

1. Simulate illuminator with positions as defined in the simulation models database:

    .. code-block:: console

        simtools-simulate-illuminator --light_source ILLN-01 \
        --telescope MSTN-04 --site North \
        --model_version 7.0.0

2. Simulate at a configurable position (1km above array center) and pointing downwards:

    .. code-block:: console

        simtools-simulate-illuminator --light_source ILLN-01 \
        --light_source_position 0. 0. 1000. \
        --light_source_pointing 0. 0. -1. \
        --telescope MSTN-15 --site North \
        --model_version 7.0.0

3. Simulate all valid pairs from the visibility table in parallel:

    .. code-block:: console

        simtools-simulate-illuminator --site North \
        --model_version 7.0.0 --simulate_all

4. Simulate all pairs for a specific illuminator only:

    .. code-block:: console

        simtools-simulate-illuminator --site North \
        --model_version 7.0.0 --simulate_all \
        --light_source ILLN-01

5. Simulate all pairs with explicit worker count:

    .. code-block:: console

        simtools-simulate-illuminator --site North \
        --model_version 7.0.0 --simulate_all \
        --max_workers 8

Command Line Arguments
----------------------
light_source (str, optional)
    Illuminator in array, e.g., ILLN-01. Required for single-pair mode.
    In multi-pair mode, used as a filter (simulate only pairs with this illuminator).
number_of_events (int, optional)
    Number of events to simulate.
flasher_photons (int, optional)
    Overwrite the model parameter flasher_photons.
telescope (str, optional)
    Telescope model name (e.g. LSTN-01, MSTN-04, ...). Required for single-pair mode.
    In multi-pair mode, used as a filter (simulate only pairs with this telescope).
site (str, required)
    Site name (North or South).
model_version (str, optional)
    Version of the simulation model.
simulate_all (flag, optional)
    Simulate all valid illuminator-telescope pairs from the visibility table.
max_workers (int, optional)
    Maximum number of parallel workers for multi-pair mode. Default: 60% of CPU cores.
    Set to 0 to use all available cores.
light_source_position (float, float, float, optional)
    Light source position (x,y,z) relative to the array center (ground coordinates) in
    m. If not set, the position from the simulation model is used.
light_source_pointing (float, float, float, optional)
    Light source pointing direction. If not set, the pointing from the simulation model is used.
"""

import logging
import sys

from simtools.application_control import build_application
from simtools.simtel.multi_illuminator_simulator import MultiIlluminatorSimulator
from simtools.simtel.simulator_light_emission import SimulatorLightEmission

_logger = logging.getLogger(__name__)


def _add_arguments(parser):
    """Register application-specific command line arguments."""
    parser.add_argument(
        "--light_source",
        help="Illuminator name, e.g. ILLN-01. Required for single-pair mode.",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--simulate_all",
        help="Simulate all valid illuminator-telescope pairs from the visibility table.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--max_workers",
        help=(
            "Maximum number of parallel workers for multi-pair mode. "
            "Default: 60%% of CPU cores. Set to 0 or negative for all cores."
        ),
        type=int,
        default=None,
        required=False,
    )
    configurable_light_source_args = parser.add_argument_group(
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
    parser.add_argument(
        "--number_of_events",
        help="Number of events to simulate",
        type=int,
        default=1,
        required=False,
    )
    parser.add_argument(
        "--flasher_photons",
        help=(
            "Override flasher photon yield. "
            "Accepts integers including scientific notation, e.g. 1e8."
        ),
        type=parser.scientific_int,
        required=False,
    )


def _simulate_single_pair(app_context):
    """Run a single illuminator-telescope simulation."""
    light_source = SimulatorLightEmission(
        light_emission_config=app_context.args,
        label=app_context.args.get("label"),
    )
    light_source.simulate()
    light_source.validate_simulations()


def _simulate_all_pairs(app_context):
    """Simulate all valid illuminator-telescope pairs in parallel."""
    simulator = MultiIlluminatorSimulator(
        config=app_context.args,
        label=app_context.args.get("label"),
        max_workers=app_context.args.get("max_workers"),
    )

    # Apply filters if specified
    illuminators = None
    telescopes = None
    if app_context.args.get("light_source"):
        illuminators = [app_context.args["light_source"]]
    if app_context.args.get("telescope"):
        telescopes = [app_context.args["telescope"]]

    results = simulator.simulate(illuminators=illuminators, telescopes=telescopes)
    summary = simulator.get_summary()

    _logger.info(
        f"Multi-illuminator simulation complete: "
        f"{summary['successful']}/{summary['total']} successful "
        f"(success rate: {summary['success_rate']:.1%})"
    )

    failed = simulator.get_failed_pairs()
    if failed:
        _logger.warning(f"Failed pairs: {failed}")

    return results


def main():
    """See CLI description."""
    app_context = build_application(
        initialization_kwargs={
            "db_config": True,
            "simulation_model": ["telescope", "model_version"],
            "require_command_line": True,
        },
    )

    if app_context.args.get("simulate_all"):
        _simulate_all_pairs(app_context)
    else:
        if not app_context.args.get("light_source"):
            sys.exit(
                "error: --light_source is required for single-pair mode. "
                "Use --simulate_all for multi-pair mode."
            )
        if not app_context.args.get("telescope"):
            sys.exit(
                "error: --telescope is required for single-pair mode. "
                "Use --simulate_all for multi-pair mode."
            )
        _simulate_single_pair(app_context)


if __name__ == "__main__":
    main()
