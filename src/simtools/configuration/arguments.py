"""Authoritative command-line argument definitions for simtools applications."""

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import astropy.units as u

import simtools.configuration.argument_helpers as helpers
import simtools.version
from simtools import constants
from simtools.configuration import defaults
from simtools.corsika.primary_particle import PrimaryParticle


def get_corsika_configuration_args():
    """Return dictionary with CORSIKA configuration parameters."""
    return {
        "primary": {
            "help": (
                "Primary particle(s) to simulate. Common names: "
                f"{', '.join(PrimaryParticle.particle_names().keys())}."
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
            "help": "Offset added to each run number.",
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
            "help": "Minimum and maximum primary energy, e.g. '10 GeV 5 TeV'.",
            "action": helpers.QuantityPairAction,
            "nargs": "+",
            "default": (3 * u.GeV, 330 * u.TeV),
        },
        "view_cone": {
            "help": "Minimum and maximum view-cone radius, e.g. '0 deg 10 deg'.",
            "type": helpers.parse_quantity_pair,
            "default": ["0 deg 0 deg"],
        },
        "core_scatter": {
            "help": "Core positions per shower and maximum scatter radius, e.g. '10 500 m'.",
            "type": helpers.parse_integer_and_quantity,
            "default": ["10 10000 m"],
        },
    },
    "CONFIGURATION_ARGS": {
        "config": {
            "help": "Application configuration file.",
            "default": None,
            "type": str,
        },
        "env_file": {
            "help": "File containing environment variables.",
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
            "help": "Directory for output files.",
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
            "help": "Output data file.",
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
            "help": "Activity identifier.",
            "type": str,
            "default": None,
        },
        "test": {
            "help": "test option for faster execution during development",
            "action": "store_true",
        },
        "label": {
            "help": "Application run label.",
        },
        "log_level": {
            "action": "store",
            "default": "info",
            "help": "Logging level.",
        },
        "log_file": {
            "help": "Log file.",
            "type": Path,
        },
        "log_file_path": {
            "help": "Directory for the generated log file.",
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
            "help": "Write build information to this file.",
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
            "help": "Show build information and exit.",
        },
    },
    "USER_ARGS": {
        "user_name": {"help": "user name", "type": str},
        "user_organization": {"help": "user organization", "type": str},
        "user_email": {"help": "user email", "type": str},
        "user_orcid": {"help": "user ORCID", "type": str},
    },
    "DB_CONFIG_ARGS": {
        "db_api_user": {"help": "Database username.", "type": str},
        "db_api_pw": {"help": "Database password.", "type": str},
        "db_api_port": {"help": "Database server port.", "type": int},
        "db_server": {"help": "Database server address.", "type": str},
        "db_api_authentication_database": {
            "help": "Authentication database name.",
            "type": str,
        },
        "db_simulation_model": {"help": "Simulation-model database name.", "type": str.strip},
        "db_simulation_model_version": {
            "help": "Simulation-model database version.",
            "type": str.strip,
        },
    },
    "SIMULATION_MODEL_ARGS": {
        "model_version": {
            "help": "Simulation production model version(s).",
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
            "help": "Simulation software workflow.",
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


@dataclass(frozen=True, init=False)
class ArgumentDefinition:
    """Definition of one command-line argument."""

    name: str
    group: str | None
    exclusive_group: str | None
    exclusive_group_required: bool
    required_unless: str | None
    kwargs: Mapping

    def __init__(
        self,
        name,
        *,
        group=None,
        exclusive_group=None,
        exclusive_group_required=False,
        required_unless=None,
        **kwargs,
    ):
        if not name or name.startswith("-"):
            raise ValueError(f"Invalid argument name: {name!r}")
        if required_unless is not None and exclusive_group is None:
            raise ValueError("required_unless requires an exclusive_group")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "group", group)
        object.__setattr__(self, "exclusive_group", exclusive_group)
        object.__setattr__(self, "exclusive_group_required", exclusive_group_required)
        object.__setattr__(self, "required_unless", required_unless)
        object.__setattr__(self, "kwargs", MappingProxyType(dict(kwargs)))

    def with_overrides(self, **overrides):
        """Return a copy with updated argparse keyword arguments."""
        return ArgumentDefinition(
            self.name,
            group=self.group,
            exclusive_group=self.exclusive_group,
            exclusive_group_required=self.exclusive_group_required,
            required_unless=self.required_unless,
            **{**self.kwargs, **overrides},
        )

    def __call__(self, **overrides):
        """Return this shared definition with optional local overrides."""
        return self.with_overrides(**overrides)

    def without_requiredness(self):
        """Return a runtime copy whose required constraints are validated later."""
        kwargs = dict(self.kwargs)
        if "required" in kwargs:
            kwargs["required"] = False
        return ArgumentDefinition(
            self.name,
            group=self.group,
            exclusive_group=self.exclusive_group,
            exclusive_group_required=False,
            required_unless=self.required_unless,
            **kwargs,
        )


class ArgumentCatalog:
    """Callable templates for one group of shared command-line arguments."""

    def __init__(self, group, definitions):
        self.group = group
        self._definitions = MappingProxyType(definitions)

    def __getattr__(self, name):
        """Return a factory for one named shared argument."""
        try:
            definition = self._definitions[name]
        except KeyError as exc:
            raise AttributeError(f"Unknown {self.group} argument: {name}") from exc

        def _argument(**overrides):
            return ArgumentDefinition(name, group=self.group, **{**definition, **overrides})

        return _argument

    def all(self):
        """Return definitions for every argument in this catalog."""
        return tuple(getattr(self, name)() for name in self._definitions)


CONFIGURATION = ArgumentCatalog("configuration", PARAMETER_DEFINITIONS["CONFIGURATION_ARGS"])
DATABASE = ArgumentCatalog("database configuration", PARAMETER_DEFINITIONS["DB_CONFIG_ARGS"])
EXECUTION = ArgumentCatalog("execution", PARAMETER_DEFINITIONS["EXECUTION_ARGS"])
OUTPUT = ArgumentCatalog("output", PARAMETER_DEFINITIONS["OUTPUT_ARGS"])
PATH = ArgumentCatalog("paths", PARAMETER_DEFINITIONS["PATH_ARGS"])
RUN_TIME = ArgumentCatalog("run time", PARAMETER_DEFINITIONS["RUN_TIME_ARGS"])
USER = ArgumentCatalog("user", PARAMETER_DEFINITIONS["USER_ARGS"])
MODEL = ArgumentCatalog("simulation model", PARAMETER_DEFINITIONS["SIMULATION_MODEL_ARGS"])
SOFTWARE = ArgumentCatalog("simulation software", PARAMETER_DEFINITIONS["SIMULATION_SOFTWARE_ARGS"])
CORSIKA = ArgumentCatalog("simulation configuration", get_corsika_configuration_args())
SHOWER = ArgumentCatalog("shower parameters", PARAMETER_DEFINITIONS["SHOWER_ARGS"])
CORSIKA_INTERACTION = ArgumentCatalog(
    "corsika configuration", PARAMETER_DEFINITIONS["CORSIKA_ARGS"]
)
SIM_TELARRAY = ArgumentCatalog("sim_telarray configuration", PARAMETER_DEFINITIONS["SIMTEL_ARGS"])
APPLICATION_ARGUMENTS = ArgumentCatalog("application", PARAMETER_DEFINITIONS["APPLICATION_ARGS"])

# Simulation model arguments
MODEL_VERSION = MODEL.model_version()
PARAMETER_VERSION = MODEL.parameter_version()
UPDATED_PARAMETER_VERSION = MODEL.updated_parameter_version()
OVERWRITE_MODEL_PARAMETERS = MODEL.overwrite_model_parameters()
SITE = MODEL.site()
TELESCOPE = MODEL.telescope()
ARRAY_LAYOUT_NAME = MODEL.array_layout_name()
ARRAY_ELEMENT_LIST = MODEL.array_element_list()
ARRAY_LAYOUT_FILE = MODEL.array_layout_file()
ARRAY_LAYOUT_PARAMETER_FILE = MODEL.array_layout_parameter_file()
PLOT_ALL_LAYOUTS = MODEL.plot_all_layouts()

# Simulation software and CORSIKA arguments
SIMULATION_SOFTWARE = SOFTWARE.simulation_software()
PRIMARY = CORSIKA.primary()
PRIMARY_ID_TYPE = CORSIKA.primary_id_type()
AZIMUTH_ANGLE = CORSIKA.azimuth_angle()
ZENITH_ANGLE = CORSIKA.zenith_angle()
SHOWERS_PER_RUN = CORSIKA.showers_per_run()
RUN_NUMBER_OFFSET = CORSIKA.run_number_offset()
RUN_NUMBER = CORSIKA.run_number()
ENERGY_RANGE = SHOWER.energy_range()
VIEW_CONE = SHOWER.view_cone()
CORE_SCATTER = SHOWER.core_scatter()
CORSIKA_HE_INTERACTION = CORSIKA_INTERACTION.corsika_he_interaction()
CORSIKA_LE_INTERACTION = CORSIKA_INTERACTION.corsika_le_interaction()

# Application-domain arguments
ALL_MODEL_VERSIONS = APPLICATION_ARGUMENTS.all_model_versions()
DATA = APPLICATION_ARGUMENTS.data()
EVENT_DATA_FILE = APPLICATION_ARGUMENTS.event_data_file()
MAX_OFFSET = APPLICATION_ARGUMENTS.max_offset()
NUMBER_OF_PHOTONS = APPLICATION_ARGUMENTS.number_of_photons()
OFF_AXIS_ANGLES = APPLICATION_ARGUMENTS.off_axis_angles()
OFFSET_STEP = APPLICATION_ARGUMENTS.offset_step()
SOURCE_DISTANCE = APPLICATION_ARGUMENTS.source_distance()
RAY_TRACING_ZENITH_ANGLE = APPLICATION_ARGUMENTS.zenith_angle()
TELESCOPE_IDS = APPLICATION_ARGUMENTS.telescope_ids()

# Paths, output, and stable simulation-configuration bundles
OUTPUT_PATH = PATH.output_path()
PATH_ARGUMENTS = PATH.all()
OUTPUT_ARGUMENTS = OUTPUT.all()
CORSIKA_CONFIGURATION_ARGUMENTS = CORSIKA.all()
SHOWER_ARGUMENTS = SHOWER.all()
CORSIKA_INTERACTION_ARGUMENTS = CORSIKA_INTERACTION.all()
SIM_TELARRAY_ARGUMENTS = SIM_TELARRAY.all()

STANDARD_ARGUMENTS = (
    *CONFIGURATION.all(),
    *EXECUTION.all(),
    *RUN_TIME.all(),
    *USER.all(),
)


def corsika_configuration_arguments(*, primary_required=True):
    """Return CORSIKA arguments with application-specific primary requiredness."""
    return tuple(
        PRIMARY(required=primary_required) if argument.name == "primary" else argument
        for argument in CORSIKA_CONFIGURATION_ARGUMENTS
    )


def _layout_argument(name, *, required_unless=None):
    """Build one argument belonging to the shared array-layout selection group."""
    return ArgumentDefinition(
        name,
        group="simulation model",
        exclusive_group="array layout",
        exclusive_group_required=True,
        required_unless=required_unless,
        **{
            "array_layout_name": ARRAY_LAYOUT_NAME,
            "array_element_list": ARRAY_ELEMENT_LIST,
            "array_layout_file": ARRAY_LAYOUT_FILE,
            "plot_all_layouts": PLOT_ALL_LAYOUTS,
        }[name].kwargs,
    )


def layout_selection_arguments(*, include_file=False, include_plot_all=False):
    """Return the standard mutually exclusive array-layout selections."""
    names = ["array_layout_name", "array_element_list"]
    if include_file:
        names.append("array_layout_file")
    if include_plot_all:
        names.append("plot_all_layouts")
    return tuple(
        _layout_argument(name, required_unless="--list_available_layouts") for name in names
    )
