#!/usr/bin/python3

r"""
Generate simulation configuration and run simulations.

Multipipe scripts will be produced as part of this application.
Allows to run array layout simulation including shower and detector simulations

The entire simulation chain, parts of it, or nothing is executed:

- shower simulations with CORSIKA only
- shower simulations with CORSIKA which are piped directly to sim_telarray using
  the sim_telarray multipipe mechanism.

.. simtools-cli-help::
   :module: simtools.applications.simulate_prod

Examples
--------

.. simtools-integration-example::
    :file: simulate_prod_proton_20_deg_north_check_output.yml

"""

from simtools.application_control import (
    build_application,
    get_application_label,
    get_module_description_line,
)
from simtools.configuration import configurator
from simtools.configuration.commandline_argument_helpers import bounded_int
from simtools.constants import CORSIKA_MAX_SEED
from simtools.production_configuration.job_grid_io import (
    SIMULATE_PROD_JOB_GRID_EXCLUSIVE_FIELDS,
    job_grid_row_to_simulate_prod_args,
    read_job_grid_row,
)
from simtools.simulator import Simulator

_INITIALIZATION_KWARGS = {
    "db_config": True,
    "simulation_model": ["site", "layout", "telescope", "model_version"],
    "simulation_configuration": {
        "software": None,
        "corsika_configuration": ["all"],
        "sim_telarray_configuration": ["all"],
    },
    "relax_required_options": ["--config", "--job_grid_file", "--job_grid_row"],
}


def _add_arguments(parser):
    """Register application-specific command line arguments."""
    parser.add_argument(
        "--corsika_file",
        help=(
            "Path to the CORSIKA input file (only relevant for simulation software 'sim_telarray')."
        ),
        type=str,
        required=False,
    )
    parser.add_argument(
        "--grid_output_path",
        help="Directory for output files for registering on the grid.",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--save_file_lists",
        help="Save lists of output and log files.",
        action="store_true",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--save_reduced_event_lists",
        help=(
            "Save reduced event lists with event data on simulated and triggered events. "
            "Saved with the same name as the sim_telarray output file (different extension). "
        ),
        action="store_true",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--corsika_seeds",
        help="Use fixed random seeds for CORSIKA for testing purposes.",
        nargs=4,
        type=bounded_int(1, CORSIKA_MAX_SEED),
        metavar=("S1", "S2", "S3", "S4"),
    )
    parser.add_argument(
        "--sequential",
        help=(
            "Enables single-core mode (as far as possible); "
            "otherwise, CORSIKA and sim_telarray run in parallel."
        ),
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--job_grid_file",
        help=(
            "Path to an ECSV job grid file produced by simtools-production-generate-grid. "
            "When provided, the selected row defines production parameters and must not be "
            "combined with manual production arguments such as '--zenith_angle'."
        ),
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--job_grid_row",
        help=(
            "1-based index of the row to read from the file given by '--job_grid_file'. "
            "Defaults to 1 (first row)."
        ),
        type=int,
        required=False,
        default=1,
    )


def _parse():
    """Parse simulate_prod arguments and resolve unambiguous job-grid row configuration."""
    config_builder = configurator.Configurator(
        label=get_application_label(__file__),
        description=get_module_description_line(__doc__),
    )
    _add_arguments(config_builder.parser)
    args_dict, db_config = config_builder.initialize(**_INITIALIZATION_KWARGS)
    _resolve_job_grid_arguments(args_dict, config_builder.config_sources, config_builder.parser)
    return args_dict, db_config


def _resolve_job_grid_arguments(args_dict, config_sources, parser):
    """Merge selected job-grid row values into args after rejecting ambiguous input."""
    explicit_keys = set(config_sources["cli"]) | set(config_sources["file"])
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


def main():
    """See CLI description."""
    app_context = build_application(
        startup_kwargs={
            "setup_io_handler": False,
        },
        parse_function=_parse,
    )

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
