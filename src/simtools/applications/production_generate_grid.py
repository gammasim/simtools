#!/usr/bin/python3

r"""Generate a simulation production grid for a wide range of simulation parameters."""

from simtools.application_control import build_application
from simtools.configuration import defaults
from simtools.configuration.commandline_argument_helpers import scientific_int
from simtools.production_configuration.simulation_jobs import (
    TOTAL_SHOWERS_ROUNDING_WARNINGS_MAX_DEFAULT,
    generate_job_grid,
)

_APPLICATION_ARG_DEFINITIONS = {
    "axis": {
        "action": "append",
        "nargs": "+",
        "help": (
            "Grid axis as NAME MIN UNIT MAX UNIT POINTS [SCALING]. Repeat for each axis. "
            "Use azimuth/zenith or ha/dec, plus offset. SCALING is linear, log, or 1/cos."
        ),
        "metavar": "AXIS",
    },
    "direction_grid_density": {
        "nargs": "+",
        "default": None,
        "help": "Direction-point density in deg^-2. Overrides POINTS for both direction axes.",
        "metavar": "DENSITY",
    },
    "local_zenith_range": {
        "nargs": "+",
        "default": None,
        "help": "Accepted local zenith range for an HA/Dec density grid, e.g. 0 deg 70 deg.",
        "metavar": "RANGE",
    },
    "local_azimuth_range": {
        "nargs": "+",
        "default": None,
        "help": (
            "Accepted local azimuth range for an HA/Dec density grid. Ranges may wrap through "
            "0 deg."
        ),
        "metavar": "RANGE",
    },
    "corsika_limits": {
        "type": str,
        "metavar": "FILE",
        "help": "ECSV lookup table of direction-dependent CORSIKA simulation limits.",
    },
    "number_of_runs": {
        "help": "Runs generated per grid point and energy interval.",
        "type": scientific_int,
        "default": None,
    },
    "total_showers": {
        "help": (
            "Target showers per grid point and energy interval; incompatible with --number_of_runs."
        ),
        "type": scientific_int,
        "default": None,
    },
    "total_showers_scaling": {
        "help": (
            "Total-shower zenith scaling. zenith_scaled applies "
            "N * exp(factor * (cos(zenith) - 1))."
        ),
        "type": str,
        "choices": ["fixed", "zenith_scaled"],
        "default": "fixed",
    },
    "zenith_angle_scaling_factor": {
        "help": "Factor in the zenith_scaled total-showers expression.",
        "type": float,
        "default": defaults.ZENITH_ANGLE_SCALING_FACTOR_DEFAULT,
    },
    "max_total_showers_rounding_warnings": {
        "help": "Maximum warnings emitted when total_showers is rounded up to complete runs.",
        "type": scientific_int,
        "default": TOTAL_SHOWERS_ROUNDING_WARNINGS_MAX_DEFAULT,
    },
    "showers_per_run_power_law": {
        "help": (
            "Scale showers per run by (E_mid / E_ref)^INDEX. Provide INDEX VALUE UNIT; "
            "E_mid is the logarithmic energy-interval midpoint."
        ),
        "nargs": 3,
        "type": str,
        "metavar": ("POWER_INDEX", "REFERENCE_ENERGY_VALUE", "REFERENCE_ENERGY_UNIT"),
        "default": None,
    },
    "showers_per_run_scaling": {
        "help": "Showers-per-run zenith scaling. cosine_zenith applies ceil(N * cos(zenith)).",
        "type": str,
        "choices": ["fixed", "cosine_zenith"],
        "default": "fixed",
    },
    "energy_max_scaling": {
        "help": (
            "Set the zenith-dependent maximum energy to VALUE * cos(zenith)^INDEX. "
            "Provide INDEX VALUE UNIT; the configured energy-range maximum remains an upper bound."
        ),
        "nargs": 3,
        "type": str,
        "metavar": ("POWER_INDEX", "REFERENCE_ENERGY_VALUE", "REFERENCE_ENERGY_UNIT"),
        "default": None,
    },
}

_INITIALIZATION_KWARGS = {
    "argument_overrides": {
        "model_version": {"required": True},
        "output_file": {
            "default": "job_grid.ecsv",
            "help": "Output ECSV production job grid.",
            "metavar": "FILE",
        },
        "run_number_offset": {
            "help": "Offset for sequential run numbers; the first run is offset + 1."
        },
        "showers_per_run": {"required": True},
        "site": {"required": True},
    },
    "common_arguments": {
        "configuration": ["config", "env_file"],
        "execution": [
            "activity_id",
            "label",
            "log_level",
            "log_file",
            "log_file_path",
            "disable_log_file",
            "export_build_info",
            "version",
            "build_info",
        ],
    },
    "db_config": True,
    "include_implicit_simulation_model_arguments": False,
    "output": ["output_file"],
    "paths": ["output_path"],
    "preserve_by_version_keys": ["array_layout_name"],
    "simulation_model": ["site", "array_layout_name", "model_version"],
    "simulation_configuration": {
        "software": None,
        "corsika_configuration": [
            "primary",
            "azimuth_angle",
            "zenith_angle",
            "showers_per_run",
            "run_number_offset",
            "energy_range",
            "view_cone",
            "core_scatter",
            "corsika_he_interaction",
            "corsika_le_interaction",
        ],
    },
}


def _add_arguments(parser):
    """Add application arguments, including mutually exclusive shower-count modes."""
    shower_count_group = parser.add_mutually_exclusive_group()
    for parameter, definition in _APPLICATION_ARG_DEFINITIONS.items():
        container = (
            shower_count_group if parameter in {"number_of_runs", "total_showers"} else parser
        )
        parser.add_parameter_from_definition(container, parameter, definition)


def main():
    """See CLI description."""
    app_context = build_application(
        add_arguments_function=_add_arguments,
        initialization_kwargs=_INITIALIZATION_KWARGS,
        startup_kwargs={"resolve_sim_software_executables": False},
    )

    generate_job_grid(
        app_context.args,
        app_context.io_handler.get_output_file(app_context.args["output_file"]),
    )


if __name__ == "__main__":
    main()
