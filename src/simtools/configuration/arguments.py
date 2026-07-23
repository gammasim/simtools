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


@dataclass(frozen=True, init=False)
class ArgumentDefinition:
    """Definition of one command-line argument."""

    name: str
    group: str | None
    exclusive_group: str | None
    exclusive_group_required: bool
    preserve_by_version: bool
    kwargs: Mapping

    def __init__(
        self,
        name,
        *,
        group=None,
        exclusive_group=None,
        exclusive_group_required=False,
        preserve_by_version=False,
        **kwargs,
    ):
        if not name or name.startswith("-"):
            raise ValueError(f"Invalid argument name: {name!r}")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "group", group)
        object.__setattr__(self, "exclusive_group", exclusive_group)
        object.__setattr__(self, "exclusive_group_required", exclusive_group_required)
        object.__setattr__(self, "preserve_by_version", preserve_by_version)
        object.__setattr__(self, "kwargs", MappingProxyType(dict(kwargs)))

    def __call__(self, **overrides):
        """Return a copy with application-local argparse overrides."""
        return ArgumentDefinition(
            self.name,
            group=self.group,
            exclusive_group=self.exclusive_group,
            exclusive_group_required=self.exclusive_group_required,
            preserve_by_version=self.preserve_by_version,
            **{**self.kwargs, **overrides},
        )

    def without_requiredness(self):
        """Return a copy whose required constraints can be validated after parsing."""
        kwargs = dict(self.kwargs)
        if "required" in kwargs:
            kwargs["required"] = False
        return ArgumentDefinition(
            self.name,
            group=self.group,
            exclusive_group=self.exclusive_group,
            exclusive_group_required=False,
            preserve_by_version=self.preserve_by_version,
            **kwargs,
        )


def _argument(name, group, **kwargs):
    """Create a shared argument definition."""
    return ArgumentDefinition(name, group=group, **kwargs)


_RUN_TIME_GROUP = "run time"
_DATABASE_CONFIGURATION_GROUP = "database configuration"
_SIMULATION_MODEL_GROUP = "simulation model"
_SIMULATION_CONFIGURATION_GROUP = "simulation configuration"
_SHOWER_PARAMETERS_GROUP = "shower parameters"
_SIM_TELARRAY_CONFIGURATION_GROUP = "sim_telarray configuration"
_CORSIKA_CONFIGURATION_GROUP = "corsika configuration"


CONFIG = _argument(
    "config",
    "configuration",
    help="Application configuration file.",
    default=None,
    type=str,
)

ENV_FILE = _argument(
    "env_file",
    "configuration",
    help="File containing environment variables.",
    default=".env",
    type=str,
)

CONFIGURATION_ARGUMENTS = (
    CONFIG,
    ENV_FILE,
)

DATA_PATH = _argument(
    "data_path",
    "paths",
    help="path pointing towards data directory",
    type=Path,
    default="./data/",
)

OUTPUT_PATH = _argument(
    "output_path",
    "paths",
    help="Directory for output files.",
    type=Path,
    default="./simtools-output/",
)

MODEL_PATH = _argument(
    "model_path",
    "paths",
    help="path pointing towards simulation model file directory",
    type=Path,
    default="./",
)

SIM_TELARRAY_PATH = _argument(
    "sim_telarray_path",
    "paths",
    help="path pointing to sim_telarray installation",
    type=Path,
)

CORSIKA_PATH = _argument(
    "corsika_path",
    "paths",
    help=f"path pointing to CORSIKA installation (default: {defaults.CORSIKA_PATH})",
    type=Path,
)

CORSIKA_INTERACTION_TABLE_PATH = _argument(
    "corsika_interaction_table_path",
    "paths",
    help=(
        "path pointing to CORSIKA interaction tables "
        f"(default: {defaults.CORSIKA_INTERACTION_TABLE_PATH})"
    ),
    type=Path,
)

PATH_ARGUMENTS = (
    DATA_PATH,
    OUTPUT_PATH,
    MODEL_PATH,
    SIM_TELARRAY_PATH,
    CORSIKA_PATH,
    CORSIKA_INTERACTION_TABLE_PATH,
)

OUTPUT_FILE = _argument(
    "output_file",
    "output",
    help="Output data file.",
    type=str,
)

OUTPUT_FILE_FORMAT = _argument(
    "output_file_format",
    "output",
    help="file format of output data",
    type=str,
    default="ecsv",
)

SKIP_OUTPUT_VALIDATION = _argument(
    "skip_output_validation",
    "output",
    help="skip output data validation against schema",
    action="store_true",
)

OUTPUT_ARGUMENTS = (
    OUTPUT_FILE,
    OUTPUT_FILE_FORMAT,
    SKIP_OUTPUT_VALIDATION,
)

RUNTIME_ENVIRONMENT_FILE = _argument(
    "runtime_environment_file",
    _RUN_TIME_GROUP,
    type=Path,
    help="Path to a standalone runtime-environment YAML file (top-level 'runtime_environment').",
    default=None,
)

APPTAINER_IMAGE = _argument(
    "apptainer_image",
    _RUN_TIME_GROUP,
    help="Apptainer image path or a dictionary mapping labels to image paths.",
    type=helpers.string_or_dict,
    default=None,
)

IGNORE_RUNTIME_ENVIRONMENT = _argument(
    "ignore_runtime_environment",
    _RUN_TIME_GROUP,
    action="store_true",
    help="Ignore the runtime environment and run the application in the current environment.",
    default=False,
)

OVERWRITE_COLLECTION_FILES = _argument(
    "overwrite_collection_files",
    _RUN_TIME_GROUP,
    action="store_true",
    help=(
        "Allow files copied by the workflow collection block to overwrite existing "
        "files with identical names."
    ),
    default=False,
)

RUN_TIME_ARGUMENTS = (
    RUNTIME_ENVIRONMENT_FILE,
    APPTAINER_IMAGE,
    IGNORE_RUNTIME_ENVIRONMENT,
    OVERWRITE_COLLECTION_FILES,
)

ACTIVITY_ID = _argument(
    "activity_id",
    "execution",
    help="Activity identifier.",
    type=str,
    default=None,
)

TEST = _argument(
    "test",
    "execution",
    help="test option for faster execution during development",
    action="store_true",
)

LABEL = _argument(
    "label",
    "execution",
    help="Application run label.",
)

LOG_LEVEL = _argument(
    "log_level",
    "execution",
    action="store",
    default="info",
    help="Logging level.",
)

LOG_FILE = _argument(
    "log_file",
    "execution",
    help="Log file.",
    type=Path,
)

LOG_FILE_PATH = _argument(
    "log_file_path",
    "execution",
    help="Directory for the generated log file.",
    type=Path,
)

DISABLE_LOG_FILE = _argument(
    "disable_log_file",
    "execution",
    action="store_true",
    help=argparse.SUPPRESS,
)

FIGURE_FORMAT = _argument(
    "figure_format",
    "execution",
    help="output figure format(s)",
    type=str,
    nargs="+",
    default=["png"],
)

EXPORT_BUILD_INFO = _argument(
    "export_build_info",
    "execution",
    help="Write build information to this file.",
    type=str,
)

IGNORE_EXISTING_PARAMETER_VERSION = _argument(
    "ignore_existing_parameter_version",
    "execution",
    action="store_true",
    help="skip checking for an existing model parameter version in the database",
)

VERSION = _argument(
    "version",
    "execution",
    action="version",
    version=f"%(prog)s {simtools.version.__version__}",
    help=argparse.SUPPRESS,
)

BUILD_INFO = _argument(
    "build_info",
    "execution",
    action=helpers.BuildInfoAction,
    build_info=f"%(prog)s {simtools.version.__version__}",
    help="Show build information and exit.",
)

EXECUTION_ARGUMENTS = (
    ACTIVITY_ID,
    TEST,
    LABEL,
    LOG_LEVEL,
    LOG_FILE,
    LOG_FILE_PATH,
    DISABLE_LOG_FILE,
    FIGURE_FORMAT,
    EXPORT_BUILD_INFO,
    IGNORE_EXISTING_PARAMETER_VERSION,
    VERSION,
    BUILD_INFO,
)

USER_NAME = _argument(
    "user_name",
    "user",
    help="user name",
    type=str,
)

USER_ORGANIZATION = _argument(
    "user_organization",
    "user",
    help="user organization",
    type=str,
)

USER_EMAIL = _argument(
    "user_email",
    "user",
    help="user email",
    type=str,
)

USER_ORCID = _argument(
    "user_orcid",
    "user",
    help="user ORCID",
    type=str,
)

USER_ARGUMENTS = (
    USER_NAME,
    USER_ORGANIZATION,
    USER_EMAIL,
    USER_ORCID,
)

DB_API_USER = _argument(
    "db_api_user",
    _DATABASE_CONFIGURATION_GROUP,
    help="Database username.",
    type=str,
)

SIMULATION_MODELS_PATH = _argument(
    "simulation_models_path",
    _DATABASE_CONFIGURATION_GROUP,
    help=(
        "Path containing simulation model files; when set, model parameters are read "
        "from files instead of MongoDB."
    ),
    type=Path,
    default=None,
)

DB_API_PW = _argument(
    "db_api_pw",
    _DATABASE_CONFIGURATION_GROUP,
    help="Database password.",
    type=str,
)

DB_API_PORT = _argument(
    "db_api_port",
    _DATABASE_CONFIGURATION_GROUP,
    help="Database server port.",
    type=int,
)

DB_SERVER = _argument(
    "db_server",
    _DATABASE_CONFIGURATION_GROUP,
    help="Database server address.",
    type=str,
)

DB_API_AUTHENTICATION_DATABASE = _argument(
    "db_api_authentication_database",
    _DATABASE_CONFIGURATION_GROUP,
    help="Authentication database name.",
    type=str,
)

DB_SIMULATION_MODEL = _argument(
    "db_simulation_model",
    _DATABASE_CONFIGURATION_GROUP,
    help="Simulation-model database name.",
    type=str.strip,
)

DB_SIMULATION_MODEL_VERSION = _argument(
    "db_simulation_model_version",
    _DATABASE_CONFIGURATION_GROUP,
    help="Simulation-model database version.",
    type=str.strip,
)

DATABASE_ARGUMENTS = (
    SIMULATION_MODELS_PATH,
    DB_API_USER,
    DB_API_PW,
    DB_API_PORT,
    DB_SERVER,
    DB_API_AUTHENTICATION_DATABASE,
    DB_SIMULATION_MODEL,
    DB_SIMULATION_MODEL_VERSION,
)

MODEL_VERSION = _argument(
    "model_version",
    _SIMULATION_MODEL_GROUP,
    help="Simulation production model version(s).",
    type=str,
    default=None,
    nargs="+",
)

PARAMETER_VERSION = _argument(
    "parameter_version",
    _SIMULATION_MODEL_GROUP,
    help="model parameter version",
    type=str,
    default=None,
)

UPDATED_PARAMETER_VERSION = _argument(
    "updated_parameter_version",
    _SIMULATION_MODEL_GROUP,
    help="updated parameter version",
    type=str,
    default=None,
)

OVERWRITE_MODEL_PARAMETERS = _argument(
    "overwrite_model_parameters",
    _SIMULATION_MODEL_GROUP,
    help="File name to overwrite model parameters from DB with provided values",
    type=str,
)

SITE = _argument(
    "site",
    _SIMULATION_MODEL_GROUP,
    help="Observatory site (e.g., North, South)",
    type=helpers.site,
)

TELESCOPE = _argument(
    "telescope",
    _SIMULATION_MODEL_GROUP,
    help="telescope model name (e.g., LSTN-01, SSTS-design, ...)",
    type=helpers.telescope,
)

TELESCOPES = _argument(
    "telescopes",
    _SIMULATION_MODEL_GROUP,
    help="list of telescopes (e.g., LSTN-01, SSTS-design, ...)",
    type=helpers.telescope,
    nargs="+",
)

ARRAY_LAYOUT_NAME = _argument(
    "array_layout_name",
    _SIMULATION_MODEL_GROUP,
    help=(
        "Array layout name(s) (e.g., CTAO-North-Alpha, LSTN-01). "
        "Telescope names are assumed as single-telescope layouts."
    ),
    nargs="+",
    type=str,
    preserve_by_version=True,
)

ARRAY_ELEMENT_LIST = _argument(
    "array_element_list",
    _SIMULATION_MODEL_GROUP,
    help="list of array elements (e.g., LSTN-01, LSTN-02, MSTN).",
    nargs="+",
    type=str,
    default=None,
)

ARRAY_LAYOUT_FILE = _argument(
    "array_layout_file",
    _SIMULATION_MODEL_GROUP,
    help="file(s) with the list of array elements (astropy table format).",
    nargs="+",
    type=str,
    default=None,
)

ARRAY_LAYOUT_PARAMETER_FILE = _argument(
    "array_layout_parameter_file",
    _SIMULATION_MODEL_GROUP,
    help="Array layout model parameter file (typically in JSON format).",
    type=str,
    default=None,
)

PLOT_ALL_LAYOUTS = _argument(
    "plot_all_layouts",
    _SIMULATION_MODEL_GROUP,
    help="plot all available layouts",
    action="store_true",
)

IGNORE_MISSING_DESIGN_MODEL = _argument(
    "ignore_missing_design_model",
    _SIMULATION_MODEL_GROUP,
    help="Ignore missing design model definition of DB",
    action="store_true",
)

SIMULATION_SOFTWARE = _argument(
    "simulation_software",
    "simulation software",
    help="Simulation software workflow.",
    type=str,
    choices=list(defaults.SIMULATION_SOFTWARE_CHOICES),
    default=defaults.SIMULATION_SOFTWARE_DEFAULT,
)

PRIMARY = _argument(
    "primary",
    _SIMULATION_CONFIGURATION_GROUP,
    help=(
        "Primary particle(s) to simulate. Common names: "
        f"{', '.join(PrimaryParticle.particle_names().keys())}."
    ),
    type=str.lower,
    action=helpers.OneOrManyAction,
    nargs="+",
    required=True,
)

PRIMARY_ID_TYPE = _argument(
    "primary_id_type",
    _SIMULATION_CONFIGURATION_GROUP,
    help="Primary particle ID type",
    type=str,
    choices=["common_name", "corsika7_id", "pdg_id"],
    default="common_name",
)

AZIMUTH_ANGLE = _argument(
    "azimuth_angle",
    _SIMULATION_CONFIGURATION_GROUP,
    help=(
        "Telescope pointing direction in azimuth. It can be in degrees between 0 and 360 "
        "or one of north, south, east or west. North is 0 degrees and the azimuth grows "
        "clockwise (East is 90 degrees)."
    ),
    type=helpers.azimuth_angle,
    action=helpers.OneOrManyAction,
    nargs="+",
    default=0 * u.deg,
)

ZENITH_ANGLE = _argument(
    "zenith_angle",
    _SIMULATION_CONFIGURATION_GROUP,
    help="Zenith angle in degrees (between 0 and 180).",
    type=helpers.zenith_angle,
    action=helpers.OneOrManyAction,
    nargs="+",
    default=20 * u.deg,
)

SHOWERS_PER_RUN = _argument(
    "showers_per_run",
    _SIMULATION_CONFIGURATION_GROUP,
    help="Baseline number of CORSIKA showers per run.",
    type=int,
)

RUN_NUMBER_OFFSET = _argument(
    "run_number_offset",
    _SIMULATION_CONFIGURATION_GROUP,
    help="Offset added to each run number.",
    type=int,
    default=0,
)

RUN_NUMBER = _argument(
    "run_number",
    _SIMULATION_CONFIGURATION_GROUP,
    help="Run number to be simulated.",
    type=int,
    default=1,
)

EVENT_NUMBER_FIRST_SHOWER = _argument(
    "event_number_first_shower",
    _SIMULATION_CONFIGURATION_GROUP,
    help="Event number of first shower",
    type=int,
    default=1,
)

CORRECT_FOR_B_FIELD_ALIGNMENT = _argument(
    "correct_for_b_field_alignment",
    _SIMULATION_CONFIGURATION_GROUP,
    help="Correct for B-field alignment",
    action="store_true",
    default=True,
)

CURVED_ATMOSPHERE_MIN_ZENITH_ANGLE = _argument(
    "curved_atmosphere_min_zenith_angle",
    _SIMULATION_CONFIGURATION_GROUP,
    help="Minimum zenith angle (deg) for using curved-atmosphere CORSIKA binaries. ",
    type=helpers.zenith_angle,
    default=defaults.CURVED_ATMOSPHERE_MIN_ZENITH_ANGLE_DEG * u.deg,
)

CORSIKA_CONFIGURATION_ARGUMENTS = (
    PRIMARY,
    PRIMARY_ID_TYPE,
    AZIMUTH_ANGLE,
    ZENITH_ANGLE,
    SHOWERS_PER_RUN,
    RUN_NUMBER_OFFSET,
    RUN_NUMBER,
    EVENT_NUMBER_FIRST_SHOWER,
    CORRECT_FOR_B_FIELD_ALIGNMENT,
    CURVED_ATMOSPHERE_MIN_ZENITH_ANGLE,
)

ESLOPE = _argument(
    "eslope",
    _SHOWER_PARAMETERS_GROUP,
    help="Slope of the energy spectrum.",
    type=float,
    default=-2.0,
)

ENERGY_RANGE = _argument(
    "energy_range",
    _SHOWER_PARAMETERS_GROUP,
    help="Minimum and maximum primary energy, e.g. '10 GeV 5 TeV'.",
    action=helpers.QuantityPairAction,
    nargs="+",
    default=(3 * u.GeV, 330 * u.TeV),
)

VIEW_CONE = _argument(
    "view_cone",
    _SHOWER_PARAMETERS_GROUP,
    help="Minimum and maximum view-cone radius, e.g. '0 deg 10 deg'.",
    type=helpers.parse_quantity_pair,
    default=["0 deg 0 deg"],
)

CORE_SCATTER = _argument(
    "core_scatter",
    _SHOWER_PARAMETERS_GROUP,
    help="Core positions per shower and maximum scatter radius, e.g. '10 500 m'.",
    type=helpers.parse_integer_and_quantity,
    default=["10 10000 m"],
)

SHOWER_ARGUMENTS = (
    ESLOPE,
    ENERGY_RANGE,
    VIEW_CONE,
    CORE_SCATTER,
)

CORSIKA_HE_INTERACTION = _argument(
    "corsika_he_interaction",
    _CORSIKA_CONFIGURATION_GROUP,
    help=(
        "High-energy interaction model for CORSIKA "
        f"(default fallback: {defaults.CORSIKA_HE_INTERACTION})."
    ),
    type=str,
    action=helpers.OneOrManyAction,
    nargs="+",
    default=None,
)

CORSIKA_LE_INTERACTION = _argument(
    "corsika_le_interaction",
    _CORSIKA_CONFIGURATION_GROUP,
    help=(
        "Low-energy interaction model for CORSIKA "
        f"(default fallback: {defaults.CORSIKA_LE_INTERACTION})."
    ),
    type=str,
    action=helpers.OneOrManyAction,
    nargs="+",
    default=None,
)

CORSIKA_HADRONIC_TRANSITION_ENERGY = _argument(
    "corsika_hadronic_transition_energy",
    _CORSIKA_CONFIGURATION_GROUP,
    help=(
        "Transition energy between the low- and high-energy CORSIKA hadronic "
        "interaction models. Unitless values are interpreted as GeV. If omitted, "
        "the CORSIKA build default is used."
    ),
    type=helpers.positive_quantity("GeV"),
    default=None,
)

CORSIKA_INTERACTION_ARGUMENTS = (
    CORSIKA_HE_INTERACTION,
    CORSIKA_LE_INTERACTION,
    CORSIKA_HADRONIC_TRANSITION_ENERGY,
)

SIM_TELARRAY_INSTRUMENT_SEED = _argument(
    "sim_telarray_instrument_seed",
    _SIM_TELARRAY_CONFIGURATION_GROUP,
    help="Random seed used for sim_telarray instrument setup.",
    type=helpers.bounded_int(1, constants.SIMTEL_MAX_SEED),
)

SIM_TELARRAY_RANDOM_INSTRUMENT_INSTANCES = _argument(
    "sim_telarray_random_instrument_instances",
    _SIM_TELARRAY_CONFIGURATION_GROUP,
    help="Number of random instrument instances initialized in sim_telarray.",
    type=helpers.bounded_int(1, 1024),
    default=1,
)

SIM_TELARRAY_SEED = _argument(
    "sim_telarray_seed",
    _SIM_TELARRAY_CONFIGURATION_GROUP,
    help=(
        "Random seed used for sim_telarray simulation. Single value: seed for event "
        "simulation. Two values: [instrument_seed, simulation_seed] (use for testing only)."
    ),
    type=helpers.bounded_int(1, constants.SIMTEL_MAX_SEED),
    nargs="+",
)

SIM_TELARRAY_SEED_FILE = _argument(
    "sim_telarray_seed_file",
    _SIM_TELARRAY_CONFIGURATION_GROUP,
    help=argparse.SUPPRESS,
    type=str,
    default="sim_telarray_instrument_seeds.txt",
)

SIM_TELARRAY_ARGUMENTS = (
    SIM_TELARRAY_INSTRUMENT_SEED,
    SIM_TELARRAY_RANDOM_INSTRUMENT_INSTANCES,
    SIM_TELARRAY_SEED,
    SIM_TELARRAY_SEED_FILE,
)

SOURCE_DISTANCE = _argument(
    "source_distance",
    "application",
    help="Source distance in km (unitless values are interpreted as km).",
    type=helpers.quantity("km"),
    default=10 * u.km,
)

RAY_TRACING_ZENITH_ANGLE = _argument(
    "zenith_angle",
    "application",
    help="Zenith angle in degrees (between 0 and 180).",
    type=helpers.zenith_angle,
    default=20 * u.deg,
)

OFF_AXIS_ANGLES = _argument(
    "off_axis_angles",
    "application",
    help="One or more off-axis angles in degrees (unitless values are interpreted as degrees).",
    type=helpers.quantity("deg"),
    nargs="+",
    default=[0.0 * u.deg],
)

NUMBER_OF_PHOTONS = _argument(
    "number_of_photons",
    "application",
    help="Number of star photons to trace (per run).",
    type=helpers.scientific_int,
    default=10000,
)

MAX_OFFSET = _argument(
    "max_offset",
    "application",
    help="Maximum offset angle in degrees (unitless values are interpreted as deg).",
    type=helpers.nonnegative_quantity("deg"),
    default=4 * u.deg,
)

OFFSET_STEP = _argument(
    "offset_step",
    "application",
    help="Offset angle step size in degrees (unitless values are interpreted as deg).",
    type=helpers.positive_quantity("deg"),
    default=0.25 * u.deg,
)

ALL_MODEL_VERSIONS = _argument(
    "all_model_versions",
    "application",
    help="Produce reports for all model versions.",
    action="store_true",
)

DATA = _argument(
    "data",
    "application",
    help="Data file name.",
    type=str,
)

EVENT_DATA_FILE = _argument(
    "event_data_file",
    "application",
    help="Event data file or glob pattern containing reduced event data.",
    type=str,
    required=True,
)

TELESCOPE_IDS = _argument(
    "telescope_ids",
    "application",
    help="Path to a file containing telescope configurations.",
    type=str,
)

STANDARD_ARGUMENTS = (
    *CONFIGURATION_ARGUMENTS,
    *EXECUTION_ARGUMENTS,
    *RUN_TIME_ARGUMENTS,
    *USER_ARGUMENTS,
)


def corsika_configuration_arguments(*, primary_required=True):
    """Return CORSIKA arguments with application-specific primary requiredness."""
    return (PRIMARY(required=primary_required), *CORSIKA_CONFIGURATION_ARGUMENTS[1:])


def _layout_argument(argument, *, required):
    """Put an argument in the shared array-layout selection group."""
    return ArgumentDefinition(
        argument.name,
        group=argument.group,
        exclusive_group="array layout",
        exclusive_group_required=required,
        preserve_by_version=argument.preserve_by_version,
        **argument.kwargs,
    )


def layout_selection_arguments(
    *, required=True, include_file=False, include_parameter_file=False, include_plot_all=False
):
    """Return the standard mutually exclusive array-layout selections."""
    arguments = [ARRAY_LAYOUT_NAME, ARRAY_ELEMENT_LIST]
    if include_file:
        arguments.append(ARRAY_LAYOUT_FILE)
    if include_parameter_file:
        arguments.append(ARRAY_LAYOUT_PARAMETER_FILE)
    if include_plot_all:
        arguments.append(PLOT_ALL_LAYOUTS)
    return tuple(_layout_argument(argument, required=required) for argument in arguments)
