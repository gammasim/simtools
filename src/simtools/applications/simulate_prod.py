#!/usr/bin/python3

r"""Generate simulation configuration and run simulations."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.configuration.argument_helpers import bounded_int
from simtools.constants import CORSIKA_MAX_SEED
from simtools.production_configuration.job_grid_io import (
    SIMULATE_PROD_JOB_GRID_EXCLUSIVE_FIELDS,
    job_grid_row_to_simulate_prod_args,
    read_job_grid_row,
)
from simtools.simulator import Simulator

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "corsika_file",
        help=(
            "Path to the CORSIKA input file (only relevant for simulation software 'sim_telarray')."
        ),
        type=str,
        required=False,
    ),
    cli.ArgumentDefinition(
        "grid_output_path",
        help="Directory for output files for registering on the grid.",
        type=str,
        required=False,
        default=None,
    ),
    cli.ArgumentDefinition(
        "save_file_lists",
        help="Save lists of output and log files.",
        action="store_true",
        required=False,
        default=False,
    ),
    cli.ArgumentDefinition(
        "save_reduced_event_lists",
        help=(
            "Save reduced event lists with event data on simulated and triggered events. "
            "Saved with the same name as the sim_telarray output file (different extension). "
        ),
        action="store_true",
        required=False,
        default=False,
    ),
    cli.ArgumentDefinition(
        "corsika_seeds",
        help="Use fixed random seeds for CORSIKA for testing purposes.",
        nargs=4,
        type=bounded_int(1, CORSIKA_MAX_SEED),
        metavar=("S1", "S2", "S3", "S4"),
    ),
    cli.ArgumentDefinition(
        "sequential",
        help=(
            "Enables single-core mode (as far as possible); "
            "otherwise, CORSIKA and sim_telarray run in parallel."
        ),
        action="store_true",
        default=False,
    ),
    cli.ArgumentDefinition(
        "job_grid_file",
        help=(
            "Path to an ECSV job grid file produced by simtools-production-generate-grid. "
            "When provided, the selected row defines production parameters and must not be "
            "combined with manual production arguments such as '--zenith_angle'."
        ),
        type=str,
        required=False,
        default=None,
    ),
    cli.ArgumentDefinition(
        "job_grid_row",
        help=(
            "1-based index of the row to read from the file given by '--job_grid_file'. "
            "Defaults to 1 (first row)."
        ),
        type=int,
        required=False,
        default=1,
    ),
)


def _resolve_job_grid_arguments(args_dict, config_sources, parser):
    """Merge selected job-grid row values into args after rejecting ambiguous input."""
    explicit_keys = set(config_sources["cli"]) | set(config_sources["yaml"])
    job_grid_row_is_explicit = "job_grid_row" in explicit_keys

    if not args_dict.get("job_grid_file"):
        if job_grid_row_is_explicit:
            parser.error("'--job_grid_row' requires '--job_grid_file'.")
        return

    conflicting_keys = sorted(explicit_keys & SIMULATE_PROD_JOB_GRID_EXCLUSIVE_FIELDS)
    if conflicting_keys:
        parser.error(
            "'--job_grid_file' cannot be combined with explicit production parameter(s): "
            + ", ".join(conflicting_keys)
        )

    job_row, metadata = read_job_grid_row(args_dict["job_grid_file"], args_dict["job_grid_row"])
    args_dict.update(job_grid_row_to_simulate_prod_args(job_row, metadata))


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        cli.MODEL_VERSION(),
        cli.OVERWRITE_MODEL_PARAMETERS(),
        cli.SITE(),
        cli.TELESCOPE(),
        *cli.layout_selection_arguments(),
        cli.SIMULATION_SOFTWARE(),
        *cli.CORSIKA_CONFIGURATION_ARGUMENTS,
        *cli.SHOWER_ARGUMENTS,
        *cli.CORSIKA_INTERACTION_ARGUMENTS,
        *cli.SIM_TELARRAY_ARGUMENTS,
        *cli.PATH_ARGUMENTS,
    ),
    database=True,
    setup_io_handler=False,
    post_parse=_resolve_job_grid_arguments,
    defer_required_validation=True,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    simulator = Simulator(label=app_context.args.get("label"))

    simulator.simulate()
    if app_context.args["save_reduced_event_lists"]:
        simulator.save_reduced_event_lists()

    simulator.validate_simulations()
    simulator.report()

    if app_context.args["save_file_lists"]:
        simulator.save_file_lists()

    if app_context.args.get("grid_output_path"):
        simulator.pack_for_register(app_context.args["grid_output_path"])


if __name__ == "__main__":
    main()
