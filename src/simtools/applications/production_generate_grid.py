#!/usr/bin/python3

r"""Generate a simulation production grid for a wide range of simulation parameters."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.configuration import defaults
from simtools.configuration.argument_helpers import scientific_int
from simtools.production_configuration.simulation_jobs import (
    TOTAL_SHOWERS_ROUNDING_WARNINGS_MAX_DEFAULT,
    generate_job_grid,
)

_GRID_ARGUMENTS = (
    cli.ArgumentDefinition(
        "axis",
        action="append",
        nargs="+",
        help=(
            "Grid axis as NAME MIN UNIT MAX UNIT POINTS [SCALING]. Repeat for each axis. "
            "Use azimuth/zenith or ha/dec, plus offset. SCALING is linear, log, or 1/cos."
        ),
        metavar="AXIS",
    ),
    cli.ArgumentDefinition(
        "direction_grid_density",
        nargs="+",
        default=None,
        help="Direction-point density in deg^-2. Overrides POINTS for both direction axes.",
        metavar="DENSITY",
    ),
    cli.ArgumentDefinition(
        "local_zenith_range",
        nargs="+",
        default=None,
        help="Accepted local zenith range for an HA/Dec density grid, e.g. 0 deg 70 deg.",
        metavar="RANGE",
    ),
    cli.ArgumentDefinition(
        "local_azimuth_range",
        nargs="+",
        default=None,
        help=(
            "Accepted local azimuth range for an HA/Dec density grid. Ranges may wrap through "
            "0 deg."
        ),
        metavar="RANGE",
    ),
    cli.ArgumentDefinition(
        "output_file",
        type=str,
        default="job_grid.ecsv",
        help="Output ECSV production job grid.",
    ),
    cli.ArgumentDefinition(
        "corsika_limits",
        type=str,
        metavar="FILE",
        help="ECSV lookup table of direction-dependent CORSIKA simulation limits.",
    ),
    cli.ArgumentDefinition(
        "number_of_runs",
        help="Runs generated per grid point and energy interval.",
        type=scientific_int,
        default=None,
    ),
    cli.ArgumentDefinition(
        "total_showers",
        help=(
            "Target showers per grid point and energy interval; incompatible with --number_of_runs."
        ),
        type=scientific_int,
        default=None,
    ),
    cli.ArgumentDefinition(
        "total_showers_scaling",
        help=(
            "Total-shower zenith scaling. zenith_scaled applies "
            "N * exp(factor * (cos(zenith) - 1))."
        ),
        type=str,
        choices=["fixed", "zenith_scaled"],
        default="fixed",
    ),
    cli.ArgumentDefinition(
        "zenith_angle_scaling_factor",
        help="Factor in the zenith_scaled total-showers expression.",
        type=float,
        default=defaults.ZENITH_ANGLE_SCALING_FACTOR_DEFAULT,
    ),
    cli.ArgumentDefinition(
        "max_total_showers_rounding_warnings",
        help="Maximum warnings emitted when total_showers is rounded up to complete runs.",
        type=scientific_int,
        default=TOTAL_SHOWERS_ROUNDING_WARNINGS_MAX_DEFAULT,
    ),
    cli.ArgumentDefinition(
        "showers_per_run_power_law",
        help=(
            "Scale showers per run by (E_mid / E_ref)^INDEX. Provide INDEX VALUE UNIT; "
            "E_mid is the logarithmic energy-interval midpoint."
        ),
        nargs=3,
        type=str,
        metavar=("POWER_INDEX", "REFERENCE_ENERGY_VALUE", "REFERENCE_ENERGY_UNIT"),
        default=None,
    ),
    cli.ArgumentDefinition(
        "showers_per_run_scaling",
        help="Showers-per-run zenith scaling. cosine_zenith applies ceil(N * cos(zenith)).",
        type=str,
        choices=["fixed", "cosine_zenith"],
        default="fixed",
    ),
    cli.ArgumentDefinition(
        "energy_max_scaling",
        help=(
            "Set the zenith-dependent maximum energy to VALUE * cos(zenith)^INDEX. "
            "Provide INDEX VALUE UNIT; the configured energy-range maximum remains an upper bound."
        ),
        nargs=3,
        type=str,
        metavar=("POWER_INDEX", "REFERENCE_ENERGY_VALUE", "REFERENCE_ENERGY_UNIT"),
        default=None,
    ),
)

APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_GRID_ARGUMENTS,
        cli.MODEL_VERSION(required=True),
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.IGNORE_MISSING_DESIGN_MODEL,
        cli.SITE(required=True),
        cli.TELESCOPE,
        *cli.layout_selection_arguments(),
        cli.SIMULATION_SOFTWARE,
        cli.PRIMARY,
        cli.PRIMARY_ID_TYPE,
        cli.AZIMUTH_ANGLE,
        cli.ZENITH_ANGLE,
        cli.SHOWERS_PER_RUN(required=True),
        cli.RUN_NUMBER_OFFSET,
        cli.RUN_NUMBER,
        cli.EVENT_NUMBER_FIRST_SHOWER,
        cli.CORRECT_FOR_B_FIELD_ALIGNMENT,
        cli.CURVED_ATMOSPHERE_MIN_ZENITH_ANGLE,
        cli.ESLOPE,
        cli.ENERGY_RANGE,
        cli.VIEW_CONE,
        cli.CORE_SCATTER,
        cli.CORSIKA_HE_INTERACTION,
        cli.CORSIKA_LE_INTERACTION,
        *cli.PATH_ARGUMENTS,
    ),
    database=True,
    resolve_sim_software_executables=False,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    generate_job_grid(
        app_context.args,
        app_context.io_handler.get_output_file(app_context.args["output_file"]),
    )


if __name__ == "__main__":
    main()
