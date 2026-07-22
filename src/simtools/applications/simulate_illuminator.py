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

6. Simulate with a specific wavelength (e.g., 355 nm):

    .. code-block:: console

        simtools-simulate-illuminator --light_source ILLN-01 \
        --telescope MSTN-04 --site North \
        --model_version 7.0.0 --wavelength 355 nm

7. Simulate with multiple wavelengths:

    .. code-block:: console

        simtools-simulate-illuminator --light_source ILLN-01 \
        --telescope MSTN-04 --site North \
        --model_version 7.0.0 --wavelength 355 nm 473 nm

8. Simulate all pairs for all wavelengths in model (no wavelength specified):

    .. code-block:: console

        simtools-simulate-illuminator --site North \
        --model_version 7.0.0 --simulate_all

9. Using a config file with specific wavelengths:

    Create a config file (e.g., illuminator_config.yml):

    .. code-block:: yaml

        site: North
        model_version: 7.0.0
        light_source: ILLN-01
        telescope: MSTN-04
        wavelength: [355, 473]

    Then run:

    .. code-block:: console

        simtools-simulate-illuminator --config illuminator_config.yml

Command Line Arguments
----------------------
light_source (str, optional)
    Illuminator in array, e.g., ILLN-01. Required for single-pair mode.
    In multi-pair mode, used as a filter (simulate only pairs with this illuminator).
number_of_events (int, optional)
    Number of events to simulate.
flasher_photons (int, optional)
    Overwrite the model parameter flasher_photons.
wavelength (float or list of float, optional)
    Wavelength(s) in nanometers (unitless values are interpreted as nm).
    Must be one of the wavelengths supported by the illuminator model.
    Multiple wavelengths can be specified (space-separated on command line,
    or as a list in config file: wavelength: [355, 473]).
    If not specified, all model wavelengths will be simulated
    Each wavelength will be validated and simulated as a separate job.
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

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.configuration.argument_helpers import positive_quantity, scientific_int
from simtools.simtel.multi_illuminator_simulator import MultiIlluminatorSimulator
from simtools.utils import general

_logger = logging.getLogger(__name__)


_ARGUMENTS = (
    cli.ArgumentDefinition(
        "light_source",
        help="Illuminator name, e.g. ILLN-01. Required for single-pair mode.",
        type=str,
        default=None,
        required=False,
    ),
    cli.ArgumentDefinition(
        "simulate_all",
        help="Simulate all valid illuminator-telescope pairs from the visibility table.",
        action="store_true",
        default=False,
    ),
    cli.ArgumentDefinition(
        "max_workers",
        help=(
            "Maximum parallel workers for multi-pair mode. Default: 60%% of CPU cores. "
            "Set to 0 or negative for all cores."
        ),
        type=int,
        default=None,
        required=False,
    ),
    cli.ArgumentDefinition(
        "light_source_position",
        group="Configurable light source position and pointing (override simulation model values)",
        help="Light source position (x,y,z) relative to the array center (ground coordinates) in m",
        metavar=("X", "Y", "Z"),
        nargs=3,
        required=False,
    ),
    cli.ArgumentDefinition(
        "light_source_pointing",
        group="Configurable light source position and pointing (override simulation model values)",
        help=(
            "Light source pointing direction "
            "(example pointing downwards: --light_source_pointing 0 0 -1)"
        ),
        metavar=("X", "Y", "Z"),
        nargs=3,
        required=False,
    ),
    cli.ArgumentDefinition(
        "number_of_events", help="Number of events to simulate", type=int, default=1, required=False
    ),
    cli.ArgumentDefinition(
        "flasher_photons",
        help=(
            "Override flasher photon yield. Accepts integers including scientific notation, "
            "e.g. 1e8."
        ),
        type=scientific_int,
        required=False,
    ),
    cli.ArgumentDefinition(
        "wavelength",
        help=(
            "Wavelength(s) in nanometers (unitless values are interpreted as nm). Values "
            "must be supported by the illuminator model (typically 266, 355, 473, or 532 "
            "nm). Specify multiple values space-separated on the command line or as a list "
            "in configuration. If omitted, simulate all model wavelengths."
        ),
        type=positive_quantity("nm"),
        nargs="+",
        required=False,
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        cli.MODEL_VERSION(),
        cli.OVERWRITE_MODEL_PARAMETERS(),
        cli.SITE(),
        cli.TELESCOPE(),
        *cli.PATH_ARGUMENTS,
    ),
    database=True,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    # Determine illuminator and telescope filters
    light_source = app_context.args.get("light_source")
    telescope = app_context.args.get("telescope")
    simulate_all = app_context.args.get("simulate_all")

    if simulate_all:
        # Multi-pair mode: filters are optional
        illuminators = [light_source] if light_source else None
        telescopes = [telescope] if telescope else None
    else:
        # Single-pair mode: both filters are required
        if not light_source or not telescope:
            sys.exit(
                "error: --light_source and --telescope are required for single-pair mode. "
                "Use --simulate_all for multi-pair mode."
            )
        illuminators = general.ensure_list(light_source)
        telescopes = general.ensure_list(telescope)

    # Always use MultiIlluminatorSimulator (single-pair is just a special case)
    simulator = MultiIlluminatorSimulator(
        config=app_context.args,
        label=app_context.args.get("label"),
        max_workers=app_context.args.get("max_workers"),
    )

    results = simulator.simulate(
        wavelengths=app_context.args.get("wavelength"),
        illuminators=illuminators,
        telescopes=telescopes,
    )

    if not results and not simulate_all:
        sys.exit(
            "error: no valid illuminator-telescope pairs found for the requested "
            "--light_source/--telescope combination."
        )


if __name__ == "__main__":
    main()
