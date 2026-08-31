#!/usr/bin/python3

"""Generate simulation configuration and run simulations."""

import argparse
import logging
import sys
from pathlib import Path

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.configuration.argument_helpers import bounded_int
from simtools.constants import CORSIKA_MAX_SEED
from simtools.corsika.build_options import get_corsika_build_report
from simtools.io.ascii_handler import write_data_to_file
from simtools.job_execution.execution import execute_jobs, options_from_args, submit_jobs
from simtools.production_configuration.job_grid_io import (
    build_simulate_prod_job_specs,
    job_grid_row_to_simulate_prod_args,
    read_job_grid,
)
from simtools.production_configuration.job_metadata import build_simulation_job_metadata
from simtools.simulator import Simulator

logger = logging.getLogger(__name__)

_JOB_METADATA_FILE = "simulate_prod_job_metadata.yml"

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "list_available_corsika_models",
        help="List interaction-model variants available in the CORSIKA installation and exit.",
        action="store_true",
        default=False,
    ),
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
        "save_corsika_output",
        help=(
            "Save CORSIKA output when piping CORSIKA directly to sim_telarray "
            "(only relevant for simulation software 'corsika_sim_telarray')."
        ),
        action="store_true",
        required=False,
        default=False,
    ),
    cli.ArgumentDefinition(
        "reduced_event_lists",
        help=(
            "Save reduced event lists with event data on simulated and triggered events. "
            "Saved with the same name as the sim_telarray output file (different extension). "
        ),
        action=argparse.BooleanOptionalAction,
        required=False,
        default=True,
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
            "When omitted with '--backend htcondor', all rows are submitted; "
            "otherwise row 1 is used."
        ),
        type=int,
        required=False,
        default=1,
    ),
    cli.ArgumentDefinition(
        "wait",
        action="store_true",
        default=False,
        help="Wait for submitted backend jobs to finish before exiting.",
    ),
)


def _list_available_corsika_models(args_dict, parser):
    """Print installed CORSIKA build variants and exit."""
    try:
        report = get_corsika_build_report(args_dict.get("corsika_path"))
    except (FileNotFoundError, PermissionError, ValueError) as exc:
        parser.error(str(exc))
    sys.stdout.write(report + "\n")
    parser.exit()


def _validate_single_interaction_models(args_dict, parser):
    """Reject interaction-model lists for a single simulation run."""
    for argument in ("corsika_he_interaction", "corsika_le_interaction"):
        if isinstance(args_dict.get(argument), list):
            parser.error(f"'--{argument}' accepts exactly one value for simulate_prod.")


def _post_parse(args_dict, config_sources, parser):
    """Apply simulate-prod validations after configuration sources are merged."""
    if args_dict["list_available_corsika_models"]:
        _list_available_corsika_models(args_dict, parser)
    _resolve_job_grid_arguments(args_dict, config_sources, parser)
    _validate_single_interaction_models(args_dict, parser)
    args_dict["_defer_simulation_dependency_validation"] = bool(
        args_dict.get("backend", "local") != "local" and args_dict.get("_job_grid_rows")
    )


def _resolve_job_grid_arguments(args_dict, config_sources, parser):
    """Merge selected job-grid row values into args after rejecting ambiguous input."""
    explicit_keys = set(config_sources["cli"]) | set(config_sources["yaml"])
    job_grid_row_is_explicit = "job_grid_row" in explicit_keys

    if not _handle_no_job_grid_file(args_dict, job_grid_row_is_explicit, parser):
        return

    rows, metadata = _load_and_validate_job_grid(args_dict, parser)
    is_parameter_scan = _check_parameter_scan(rows)

    selected_row = None
    if args_dict.get("backend", "local") == "local" or job_grid_row_is_explicit:
        selected_row, rows = _select_rows(
            args_dict, rows, is_parameter_scan, job_grid_row_is_explicit, parser
        )

    if args_dict.get("backend", "local") != "local":
        args_dict["_job_grid_rows"] = rows
        args_dict["_job_grid_metadata"] = metadata
        return

    _finalize_args_dict(args_dict, rows, metadata, selected_row, parser)


def _handle_no_job_grid_file(args_dict, job_grid_row_is_explicit, parser):
    """Handle case where no job grid file is provided."""
    if args_dict.get("job_grid_file"):
        return True
    if job_grid_row_is_explicit:
        parser.error("'--job_grid_row' requires '--job_grid_file'.")
    _validate_layout_selection(args_dict, parser)
    _validate_simulation_arguments(args_dict, parser)
    return False


def _load_and_validate_job_grid(args_dict, parser):
    """Load job grid and validate its contents."""
    rows, metadata = read_job_grid(args_dict["job_grid_file"])
    if not rows:
        parser.error("Job grid contains no rows to process.")
    _validate_array_layout_names(rows, parser)
    return rows, metadata


def _validate_array_layout_names(rows, parser):
    """Validate that all rows have array_layout_name."""
    missing_layout_rows = [
        index + 1 for index, row in enumerate(rows) if not row.get("array_layout_name")
    ]
    if missing_layout_rows:
        parser.error(
            "Job grid row(s) missing array_layout_name: " + ", ".join(map(str, missing_layout_rows))
        )


def _check_parameter_scan(rows):
    """Check if the job grid is a parameter scan grid."""
    has_model_parameter_set = any(row.get("model_parameter_set") for row in rows)
    logger.info(f"has_model_parameter_set: {has_model_parameter_set}")
    return has_model_parameter_set


def _select_rows(args_dict, rows, is_parameter_scan, job_grid_row_is_explicit, parser):
    """Select rows based on job_grid_row and grid type."""
    row_index = args_dict.get("job_grid_row")
    if job_grid_row_is_explicit and row_index is not None:
        return _select_explicit_row(rows, row_index, parser), rows
    if is_parameter_scan:
        return None, rows
    if row_index is not None:
        return _select_implicit_row(rows, row_index, parser), rows
    return rows[0], [rows[0]]


def _select_explicit_row(rows, row_index, parser):
    """Select a row by explicit index."""
    if row_index < 1 or row_index > len(rows):
        parser.error(f"Row index {row_index} is out of range for a grid with {len(rows)} row(s).")
    return rows[row_index - 1]


def _select_implicit_row(rows, row_index, parser):
    """Select a row by implicit index."""
    if row_index < 1 or row_index > len(rows):
        parser.error(f"Row index {row_index} is out of range for a grid with {len(rows)} row(s).")
    return rows[row_index - 1]


def _finalize_args_dict(args_dict, rows, metadata, selected_row, parser):
    """Finalize the args_dict based on selected rows."""
    if selected_row is None:
        logger.info(f"Setting _job_grid_rows to {len(rows)} rows for parameter scan")
        args_dict["_job_grid_rows"] = rows
        args_dict["_job_grid_metadata"] = metadata
    else:
        args_dict.update(job_grid_row_to_simulate_prod_args(selected_row, metadata))
        _validate_simulation_arguments(args_dict, parser)


def _validate_layout_selection(args_dict, parser):
    """Require a direct array-layout selection when no job grid supplies one."""
    if args_dict.get("array_layout_name"):
        return
    parser.error("the following argument is required: --array_layout_name")


def _execute_job_grid(args_dict):
    """Execute all selected production-grid rows through the configured backend."""
    job_specs = build_simulate_prod_job_specs(
        args_dict,
        args_dict["_job_grid_rows"],
        APPLICATION.build_parser(),
        args_dict.get("_job_grid_metadata"),
    )
    options = options_from_args(
        args_dict,
        work_dir=Path(args_dict["output_path"]),
    )
    if args_dict.get("wait", False):
        execute_jobs(job_specs, options)
    else:
        submit_jobs(job_specs, options)


def _validate_simulation_arguments(args_dict, parser):
    """Validate requirements that depend on the selected simulation software."""
    if "corsika" in args_dict["simulation_software"] and not args_dict.get("primary"):
        parser.error("the following argument is required for CORSIKA: --primary")


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        *cli.BACKEND_ARGUMENTS,
        cli.MODEL_VERSION,
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.SITE,
        cli.ARRAY_LAYOUT_NAME(required=False),
        cli.SIMULATION_SOFTWARE,
        *cli.corsika_configuration_arguments(primary_required=False),
        *cli.SHOWER_ARGUMENTS,
        *cli.CORSIKA_INTERACTION_ARGUMENTS,
        *cli.SIM_TELARRAY_ARGUMENTS,
        *cli.OUTPUT_PATH_ARGUMENTS,
        *cli.SIM_TELARRAY_PATH_ARGUMENTS,
        *cli.CORSIKA_PATH_ARGUMENTS,
    ),
    database=True,
    setup_io_handler=False,
    validate_simulation_dependencies=True,
    post_parse=_post_parse,
    defer_required_validation=True,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    # Check if we have a job grid file with parameter scan
    if app_context.args.get("job_grid_file") and app_context.args.get("_job_grid_rows") is not None:
        _execute_job_grid(app_context.args)
        return
    if app_context.args.get("job_grid_file"):
        # Job grid file but not parameter scan - this shouldn't happen with our logic
        # Fall back to single row processing
        logger.warning("Job grid file detected but _job_grid_rows not set")

    # Construct label with scan_label if present
    label = app_context.args.get("label")
    scan_label = app_context.args.get("scan_label")
    if scan_label:
        label = f"{label}_{scan_label}" if label else f"simulate-prod_{scan_label}"

    simulator = Simulator(label=label)

    simulator.simulate()
    if app_context.args["reduced_event_lists"]:
        simulator.save_reduced_event_lists()

    simulator.validate_simulations()
    simulator.report()

    if app_context.args["save_file_lists"]:
        simulator.save_file_lists()

    if app_context.args.get("grid_output_path"):
        grid_output_path = Path(app_context.args["grid_output_path"])
        simulator.pack_for_register(grid_output_path)
        write_data_to_file(
            build_simulation_job_metadata(app_context.args, simulator),
            grid_output_path / _JOB_METADATA_FILE,
        )


if __name__ == "__main__":
    main()
