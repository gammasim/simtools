"""Shared command-line parameter definitions for simtools applications."""

import argparse
from pathlib import Path

import astropy.units as u

import simtools.configuration.commandline_argument_helpers as helpers
import simtools.version
from simtools import constants
from simtools.configuration import defaults
from simtools.corsika.primary_particle import PrimaryParticle


def get_corsika_configuration_args():
    """Return dictionary with CORSIKA configuration parameters."""
    return {
        "primary": {
            "help": (
                "Primary particle to simulate. "
                "(choices for common names: "
                f"{', '.join(PrimaryParticle.particle_names().keys())}; "
                "use '--primary_id_type' to use other particle ID types)."
            ),
            "type": str.lower,
            "action": helpers.OneOrManyAction,
            "nargs": "+",
            "required": True,
        },
        "primary_id_type": {
            "help": "Primary particle ID type",
            "type": str,
            "choices": ["common_name", "corsika7_id", "pdg_id"],
            "default": "common_name",
        },
        "azimuth_angle": {
            "help": (
                "Telescope pointing direction in azimuth. "
                "It can be in degrees between 0 and 360 or one of north, south, east or west. "
                "North is 0 degrees and the azimuth grows clockwise (East is 90 degrees)."
            ),
            "type": helpers.azimuth_angle,
            "action": helpers.OneOrManyAction,
            "nargs": "+",
            "default": 0 * u.deg,
        },
        "zenith_angle": {
            "help": "Zenith angle in degrees (between 0 and 180).",
            "type": helpers.zenith_angle,
            "action": helpers.OneOrManyAction,
            "nargs": "+",
            "default": 20 * u.deg,
        },
        "showers_per_run": {
            "help": "Baseline number of CORSIKA showers per run.",
            "type": int,
        },
        "run_number_offset": {
            "help": "Offset added to run number when executing a simulation.",
            "type": int,
            "default": 0,
        },
        "run_number": {
            "help": "Run number to be simulated.",
            "type": int,
            "default": 1,
        },
        "event_number_first_shower": {
            "help": "Event number of first shower",
            "type": int,
            "default": 1,
        },
        "correct_for_b_field_alignment": {
            "help": "Correct for B-field alignment",
            "action": "store_true",
            "default": True,
        },
        "curved_atmosphere_min_zenith_angle": {
            "help": "Minimum zenith angle (deg) for using curved-atmosphere CORSIKA binaries. ",
            "type": helpers.zenith_angle,
            "default": defaults.CURVED_ATMOSPHERE_MIN_ZENITH_ANGLE_DEG * u.deg,
        },
    }


PARAMETER_DEFINITIONS = {
    "SHOWER_ARGS": {
        "eslope": {
            "help": "Slope of the energy spectrum.",
            "type": float,
            "default": -2.0,
        },
        "energy_range": {
            "help": ("Energy range of the primary particle (min/max, e.g., '10 GeV 5 TeV')."),
            "action": helpers.QuantityPairAction,
            "nargs": "+",
            "default": (3 * u.GeV, 330 * u.TeV),
        },
        "view_cone": {
            "help": (
                "View cone radius for primary arrival directions "
                "(min/max value, e.g. '0 deg 10 deg')."
            ),
            "type": helpers.parse_quantity_pair,
            "default": ["0 deg 0 deg"],
        },
        "core_scatter": {
            "help": "Scatter radius for shower cores (number of use; scatter radius).",
            "type": helpers.parse_integer_and_quantity,
            "default": ["10 10000 m"],
        },
    },
    "CONFIGURATION_ARGS": {
        "config": {
            "help": "simtools configuration file",
            "default": None,
            "type": str,
        },
        "env_file": {
            "help": "file with environment variables",
            "default": ".env",
            "type": str,
        },
    },
    "PATH_ARGS": {
        "data_path": {
            "help": "path pointing towards data directory",
            "type": Path,
            "default": "./data/",
        },
        "output_path": {
            "help": "path pointing towards output directory",
            "type": Path,
            "default": "./simtools-output/",
        },
        "model_path": {
            "help": "path pointing towards simulation model file directory",
            "type": Path,
            "default": "./",
        },
        "sim_telarray_path": {
            "help": "path pointing to sim_telarray installation",
            "type": Path,
        },
        "corsika_path": {
            "help": f"path pointing to CORSIKA installation (default: {defaults.CORSIKA_PATH})",
            "type": Path,
        },
        "corsika_interaction_table_path": {
            "help": (
                "path pointing to CORSIKA interaction tables "
                f"(default: {defaults.CORSIKA_INTERACTION_TABLE_PATH})"
            ),
            "type": Path,
        },
    },
    "OUTPUT_ARGS": {
        "output_file": {
            "help": "output data file",
            "type": str,
        },
        "output_file_format": {
            "help": "file format of output data",
            "type": str,
            "default": "ecsv",
        },
        "skip_output_validation": {
            "help": "skip output data validation against schema",
            "action": "store_true",
        },
    },
    "RUN_TIME_ARGS": {
        "runtime_environment_file": {
            "type": Path,
            "help": (
                "Path to a standalone runtime-environment YAML file "
                "(top-level 'runtime_environment')."
            ),
            "default": None,
        },
        "apptainer_image": {
            "help": "Apptainer image path or a dictionary mapping labels to image paths.",
            "type": helpers.string_or_dict,
            "default": None,
        },
        "ignore_runtime_environment": {
            "action": "store_true",
            "help": (
                "Ignore the runtime environment and run the application in the current environment."
            ),
            "default": False,
        },
        "overwrite_collection_files": {
            "action": "store_true",
            "help": (
                "Allow files copied by the workflow collection block to overwrite existing "
                "files with identical names."
            ),
            "default": False,
        },
    },
    "EXECUTION_ARGS": {
        "activity_id": {
            "help": "activity identifier",
            "type": str,
            "default": None,
        },
        "test": {
            "help": "test option for faster execution during development",
            "action": "store_true",
        },
        "label": {
            "help": "job label",
        },
        "log_level": {
            "action": "store",
            "default": "info",
            "help": "log level to print",
        },
        "log_file": {
            "help": "log file path",
            "type": Path,
        },
        "log_file_path": {
            "help": "path pointing towards log directory",
            "type": Path,
        },
        "disable_log_file": {
            "action": "store_true",
            "help": argparse.SUPPRESS,
        },
        "figure_format": {
            "help": "output figure format(s)",
            "type": str,
            "nargs": "+",
            "default": ["png"],
        },
        "export_build_info": {
            "help": "export build information to file",
            "type": str,
        },
        "ignore_existing_parameter_version": {
            "action": "store_true",
            "help": "skip checking for an existing model parameter version in the database",
        },
        "version": {
            "action": "version",
            "version": f"%(prog)s {simtools.version.__version__}",
            "help": argparse.SUPPRESS,
        },
        "build_info": {
            "action": helpers.BuildInfoAction,
            "build_info": f"%(prog)s {simtools.version.__version__}",
            "help": "show build information and exit",
        },
    },
    "USER_ARGS": {
        "user_name": {"help": "user name", "type": str},
        "user_organization": {"help": "user organization", "type": str},
        "user_email": {"help": "user email", "type": str},
        "user_orcid": {"help": "user ORCID", "type": str},
    },
    "DB_CONFIG_ARGS": {
        "simulation_models_path": {
            "help": (
                "path containing simulation model files; when set, "
                "model parameters are read from files instead of MongoDB"
            ),
            "type": Path,
            "default": None,
        },
        "db_api_user": {"help": "database user", "type": str},
        "db_api_pw": {"help": "database password", "type": str},
        "db_api_port": {"help": "database port", "type": int},
        "db_server": {"help": "database server address", "type": str},
        "db_api_authentication_database": {
            "help": "database with user info (optional)",
            "type": str,
        },
        "db_simulation_model": {"help": "name of simulation model database", "type": str.strip},
        "db_simulation_model_version": {
            "help": "version of simulation model database",
            "type": str.strip,
        },
    },
    "SIMULATION_MODEL_ARGS": {
        "model_version": {
            "help": "Simulation production model version",
            "type": str,
            "default": None,
            "nargs": "+",
        },
        "parameter_version": {
            "help": "model parameter version",
            "type": str,
            "default": None,
        },
        "updated_parameter_version": {
            "help": "updated parameter version",
            "type": str,
            "default": None,
        },
        "overwrite_model_parameters": {
            "help": "File name to overwrite model parameters from DB with provided values",
            "type": str,
        },
        "site": {
            "help": "Observatory site (e.g., North, South)",
            "type": helpers.site,
        },
        "telescope": {
            "help": "telescope model name (e.g., LSTN-01, SSTS-design, ...)",
            "type": helpers.telescope,
        },
        "telescopes": {
            "help": "list of telescopes (e.g., LSTN-01, SSTS-design, ...)",
            "type": helpers.telescope,
            "nargs": "+",
        },
        "array_layout_name": {
            "help": (
                "Array layout name(s) (e.g., CTAO-North-Alpha, LSTN-01). "
                "Telescope names are assumed as single-telescope layouts."
            ),
            "nargs": "+",
            "type": str,
        },
        "array_element_list": {
            "help": "list of array elements (e.g., LSTN-01, LSTN-02, MSTN).",
            "nargs": "+",
            "type": str,
            "default": None,
        },
        "array_layout_file": {
            "help": "file(s) with the list of array elements (astropy table format).",
            "nargs": "+",
            "type": str,
            "default": None,
        },
        "array_layout_parameter_file": {
            "help": "Array layout model parameter file (typically in JSON format).",
            "type": str,
            "default": None,
        },
        "plot_all_layouts": {
            "help": "plot all available layouts",
            "action": "store_true",
        },
        "ignore_missing_design_model": {
            "help": "Ignore missing design model definition of DB",
            "action": "store_true",
        },
    },
    "SIMTEL_ARGS": {
        "sim_telarray_instrument_seed": {
            "help": "Random seed used for sim_telarray instrument setup.",
            "type": helpers.bounded_int(1, constants.SIMTEL_MAX_SEED),
        },
        "sim_telarray_random_instrument_instances": {
            "help": "Number of random instrument instances initialized in sim_telarray.",
            "type": helpers.bounded_int(1, 1024),
            "default": 1,
        },
        "sim_telarray_seed": {
            "help": (
                "Random seed used for sim_telarray simulation. "
                "Single value: seed for event simulation. "
                "Two values: [instrument_seed, simulation_seed] (use for testing only)."
            ),
            "type": helpers.bounded_int(1, constants.SIMTEL_MAX_SEED),
            "nargs": "+",
        },
        "sim_telarray_seed_file": {
            "help": argparse.SUPPRESS,
            "type": str,
            "default": "sim_telarray_instrument_seeds.txt",
        },
    },
    "SIMULATION_SOFTWARE_ARGS": {
        "simulation_software": {
            "help": "Simulation software steps.",
            "type": str,
            "choices": list(defaults.SIMULATION_SOFTWARE_CHOICES),
            "default": defaults.SIMULATION_SOFTWARE_DEFAULT,
        }
    },
    "CORSIKA_ARGS": {
        "corsika_he_interaction": {
            "help": (
                "High-energy interaction model for CORSIKA "
                f"(default fallback: {defaults.CORSIKA_HE_INTERACTION})."
            ),
            "type": str,
            "action": helpers.OneOrManyAction,
            "nargs": "+",
            "default": None,
        },
        "corsika_le_interaction": {
            "help": (
                "Low-energy interaction model for CORSIKA "
                f"(default fallback: {defaults.CORSIKA_LE_INTERACTION})."
            ),
            "type": str,
            "action": helpers.OneOrManyAction,
            "nargs": "+",
            "default": None,
        },
    },
    "APPLICATION_ARGS": {
        "source_distance": {
            "help": "Source distance in km (unitless values are interpreted as km).",
            "type": helpers.quantity("km"),
            "default": 10 * u.km,
        },
        "zenith_angle": {
            "help": "Zenith angle in degrees (between 0 and 180).",
            "type": helpers.zenith_angle,
            "default": 20 * u.deg,
        },
        "off_axis_angles": {
            "help": (
                "One or more off-axis angles in degrees "
                "(unitless values are interpreted as degrees)."
            ),
            "type": helpers.quantity("deg"),
            "nargs": "+",
            "default": [0.0 * u.deg],
        },
        "number_of_photons": {
            "help": "Number of star photons to trace (per run).",
            "type": helpers.scientific_int,
            "default": 10000,
        },
        "max_offset": {
            "help": "Maximum offset angle in degrees (unitless values are interpreted as deg).",
            "type": helpers.nonnegative_quantity("deg"),
            "default": 4 * u.deg,
        },
        "offset_step": {
            "help": "Offset angle step size in degrees (unitless values are interpreted as deg).",
            "type": helpers.positive_quantity("deg"),
            "default": 0.25 * u.deg,
        },
        "all_model_versions": {
            "help": "Produce reports for all model versions.",
            "action": "store_true",
        },
        "data": {
            "help": "Data file name.",
            "type": str,
        },
        "event_data_file": {
            "help": "Event data file or glob pattern containing reduced event data.",
            "type": str,
            "required": True,
        },
        "telescope_ids": {
            "help": "Path to a file containing telescope configurations.",
            "type": str,
        },
    },
}
