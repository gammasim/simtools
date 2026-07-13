#!/usr/bin/python3

r"""
Generate a simulation production grid for a wide range of simulation parameters.

Expands a production definition consistent of configuration axes, energy ranges,
and run statistics into a grid of executable simulation jobs.

Possible axes include:

* Particle type (gamma, proton, electron, etc.)
* Simulation model version
* Interaction models
* Pointing directions (azimuth, zenith, hour angle, declination)

The generated grid is written to an ECSV file, which can be used as input for
local production execution or workload-management submission tools.

Different levels of night-sky background (NSB) can be configured through the
production model version.

.. simtools-cli-help::
   :module: simtools.applications.production_generate_grid

Examples
--------

.. simtools-integration-example::
    :file: production_generate_grid_horizontal_explicit.yml
    :show-command:

.. simtools-integration-example::
    :file: production_generate_grid_horizontal.yml
    :show-command:

.. simtools-integration-example::
    :file: production_generate_grid_horizontal_density.yml
    :show-command:

.. simtools-integration-example::
    :file: production_generate_grid_ha_dec_density.yml
    :show-command:

"""

from simtools.application_control import build_application
from simtools.configuration import defaults
from simtools.configuration.commandline_argument_helpers import scientific_int
from simtools.production_configuration.job_grid_io import serialize_job_grid
from simtools.production_configuration.simulation_jobs import (
    TOTAL_SHOWERS_ROUNDING_WARNINGS_MAX_DEFAULT,
    build_job_grid_metadata,
    build_simulation_jobs,
    renumber_job_rows,
)

_APPLICATION_ARG_DEFINITIONS = {
    "axis": {
        "action": "append",
        "nargs": "+",
        "required": False,
        "help": (
            "Compact axis definition: --axis <name> <min> <unit> <max> <unit> <binning> "
            "[scaling]. May be repeated. Supported axes: azimuth, zenith, ha, dec, offset. "
            "Options for scaling are: linear, log, 1/cos"
        ),
    },
    "direction_grid_density": {
        "nargs": "+",
        "required": False,
        "default": None,
        "help": (
            "Direction-grid density in 1/deg^2. If set, direction-axis binning is "
            "derived from axis ranges and this density. With HA/Dec axes, use "
            "local_zenith_range/local_azimuth_range to filter generated points."
        ),
    },
    "local_zenith_range": {
        "nargs": "+",
        "required": False,
        "default": None,
        "help": (
            "Local zenith range (quantity pair) used to filter HA/Dec density points, "
            "for example: --local_zenith_range 0 deg 70 deg"
        ),
    },
    "local_azimuth_range": {
        "nargs": "+",
        "required": False,
        "default": None,
        "help": (
            "Local azimuth range (quantity pair) used to filter HA/Dec density points, "
            "for example: --local_azimuth_range 300 deg 60 deg"
        ),
    },
    "output_file": {
        "type": str,
        "default": "job_grid.ecsv",
        "help": "Output file for the generated executable job grid.",
    },
    "corsika_limits": {
        "type": str,
        "required": False,
        "help": "Path to the lookup table for simulation limits.",
    },
    "number_of_runs": {
        "help": "Number of runs to be simulated for each production grid point and energy range..",
        "type": scientific_int,
        "required": False,
        "default": None,
    },
    "total_showers": {
        "help": "Total number of showers to simulate per production grid point and energy range.",
        "type": scientific_int,
        "required": False,
        "default": None,
    },
    "total_showers_scaling": {
        "help": "Scaling mode for total showers.",
        "type": str,
        "choices": ["fixed", "zenith_scaled"],
        "required": False,
        "default": "fixed",
    },
    "zenith_angle_scaling_factor": {
        "help": (
            "Scaling factor for zenith-dependent total_showers scaling. "
            "Used only when --total_showers_scaling is 'zenith_scaled'."
        ),
        "type": float,
        "required": False,
        "default": defaults.ZENITH_ANGLE_SCALING_FACTOR_DEFAULT,
    },
    "max_total_showers_rounding_warnings": {
        "help": (
            "Maximum number of per-point warnings emitted when total_showers is "
            "rounded up to keep equal showers per run."
        ),
        "type": scientific_int,
        "required": False,
        "default": TOTAL_SHOWERS_ROUNDING_WARNINGS_MAX_DEFAULT,
    },
    "showers_per_run_power_law": {
        "help": (
            "Scale showers_per_run by (E_mid / E_ref) ** power_index using the bin midpoint: "
            "<power_index> <reference_energy_value> <reference_energy_unit> "
            "(for example: --showers_per_run_power_law -2.0 1 TeV)."
        ),
        "nargs": 3,
        "type": str,
        "metavar": ("POWER_INDEX", "REFERENCE_ENERGY_VALUE", "REFERENCE_ENERGY_UNIT"),
        "required": False,
        "default": None,
    },
    "showers_per_run_scaling": {
        "help": (
            "Zenith-angle scaling mode for showers_per_run: "
            "'fixed' keeps the baseline value, "
            "'cosine_zenith' applies showers_per_run * cos(zenith_angle)."
        ),
        "type": str,
        "choices": ["fixed", "cosine_zenith"],
        "required": False,
        "default": "fixed",
    },
    "energy_max_scaling": {
        "help": (
            "Scale max energy with zenith angle as "
            "energy_max_scaling_reference * cos(zenith_angle) ** power_index. "
            "Provide: <power_index> <reference_energy_value> <reference_energy_unit> "
            "(for example: --energy_max_scaling -2.5 300 TeV). "
            "Max energy is limited by the configured energy_range."
        ),
        "nargs": 3,
        "type": str,
        "metavar": ("POWER_INDEX", "REFERENCE_ENERGY_VALUE", "REFERENCE_ENERGY_UNIT"),
        "required": False,
        "default": None,
    },
}


def main():
    """See CLI description."""
    app_context = build_application(
        application_argument_definitions=_APPLICATION_ARG_DEFINITIONS,
        initialization_kwargs={
            "db_config": True,
            "preserve_by_version_keys": ["array_layout_name"],
            "simulation_model": ["site", "layout", "telescope", "model_version"],
            "simulation_configuration": {"software": None, "corsika_configuration": ["all"]},
        },
    )

    job_rows = build_simulation_jobs(app_context.args)
    serialize_job_grid(
        job_rows=renumber_job_rows(job_rows, app_context.args.get("run_number_offset", 0)),
        output_file=app_context.io_handler.get_output_file(app_context.args["output_file"]),
        metadata=build_job_grid_metadata(app_context.args),
    )


if __name__ == "__main__":
    main()
