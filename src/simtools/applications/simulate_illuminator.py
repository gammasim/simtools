#!/usr/bin/python3

"""Simulate illuminator (distant calibration light source)."""

import logging
import sys

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.configuration.argument_helpers import positive_quantity, scientific_int
from simtools.simtel.multi_illuminator_simulator import MultiIlluminatorSimulator

_logger = logging.getLogger(__name__)

_LIGHT_SOURCE_GEOMETRY_GROUP = (
    "Configurable light source position and pointing (override simulation model values)"
)

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "light_source",
        help="Illuminator name, e.g. ILLN-01. Required for single-pair mode.",
        type=str,
        default=None,
    ),
    cli.ArgumentDefinition(
        "simulate_all",
        help="Simulate all valid illuminator-telescope pairs from the visibility table.",
        action="store_true",
        default=False,
    ),
    cli.MAX_WORKERS,
    cli.ArgumentDefinition(
        "light_source_position",
        group=_LIGHT_SOURCE_GEOMETRY_GROUP,
        help="Light source position (x,y,z) relative to the array center (ground coordinates) in m",
        metavar=("X", "Y", "Z"),
        type=float,
        nargs=3,
    ),
    cli.ArgumentDefinition(
        "light_source_pointing",
        group=_LIGHT_SOURCE_GEOMETRY_GROUP,
        help=(
            "Light source pointing direction "
            "(example pointing downwards: --light_source_pointing 0 0 -1)"
        ),
        metavar=("X", "Y", "Z"),
        type=float,
        nargs=3,
    ),
    cli.ArgumentDefinition(
        "number_of_events", help="Number of events to simulate", type=int, default=1
    ),
    cli.ArgumentDefinition(
        "flasher_photons",
        help=(
            "Override flasher photon yield. Accepts integers including scientific notation, "
            "e.g. 1e8."
        ),
        type=scientific_int,
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
    ),
)


def _validate_pair_selection(args_dict, _config_sources, parser):
    """Require one illuminator and one telescope unless all pairs are simulated."""
    if args_dict.get("simulate_all"):
        return
    if not args_dict.get("light_source") or not args_dict.get("telescope"):
        parser.error(
            "--light_source and --telescope are required for single-pair mode. "
            "Use --simulate_all for multi-pair mode."
        )


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    model_repository=True,
    arguments=(
        *_ARGUMENTS,
        *cli.BACKEND_ARGUMENTS,
        cli.MODEL_VERSION,
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.SITE,
        cli.TELESCOPE,
        *cli.OUTPUT_PATH_ARGUMENTS,
        *cli.SIM_TELARRAY_PATH_ARGUMENTS,
    ),
    validate_simulation_dependencies=True,
    post_parse=_validate_pair_selection,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    # Filters are optional with --simulate_all; single-pair mode requires both (post-parse).
    light_source = app_context.args.get("light_source")
    telescope = app_context.args.get("telescope")
    illuminators = [light_source] if light_source else None
    telescopes = [telescope] if telescope else None

    # Always use MultiIlluminatorSimulator (single-pair is just a special case)
    simulator = MultiIlluminatorSimulator(
        config=app_context.args,
        label=app_context.args.get("label"),
        max_workers=app_context.args.get("max_workers"),
        backend=app_context.args.get("backend", "local"),
        backend_config=app_context.args.get("backend_config"),
        model_reader=app_context.model_reader,
    )

    results = simulator.simulate(
        wavelengths=app_context.args.get("wavelength"),
        illuminators=illuminators,
        telescopes=telescopes,
    )

    failed_results = [result for result in results if not result["success"]]
    if failed_results:
        failures = "; ".join(
            f"{result['illuminator']} -> {result['telescope']}: {result['error']}"
            for result in failed_results
        )
        sys.exit(f"error: illuminator simulation failed: {failures}")

    if not results:
        visibility_count = simulator.visibility.n_valid_pairs
        light_source_filter = light_source or "all"
        telescope_filter = telescope or "all"
        sys.exit(
            "error: no illuminator-telescope pairs were scheduled for "
            f"light_source={light_source_filter}, telescope={telescope_filter}; "
            f"the visibility table contains {visibility_count} valid pairs."
        )


if __name__ == "__main__":
    main()
