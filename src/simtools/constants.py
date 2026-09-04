"""Project wide constants."""

import os
from importlib.resources import files
from pathlib import Path

# Schema path
SCHEMA_PATH = files("simtools") / "schemas"
# Schema URL
SCHEMA_URL = "https://raw.githubusercontent.com/gammasim/simtools/main/src/simtools/schemas"
# URL for files in the main branch of the GitHub repository
GITHUB_BLOB_BASE = "https://github.com/gammasim/simtools/blob/main/"
# Path to metadata jsonschema
METADATA_JSON_SCHEMA = SCHEMA_PATH / "metadata.metaschema.yml"
# Path to plotting configuration json schema
PLOT_CONFIG_SCHEMA = SCHEMA_PATH / "plot_configuration.metaschema.yml"
# Path to MongoDB configuration schema
DATABASE_SCHEMA = SCHEMA_PATH / "database.schema.yml"
# Path to run time environment json schema
RUN_TIME_ENVIRONMENT_SCHEMA = SCHEMA_PATH / "run_time_environment.schema.yml"
# Path to model parameter metaschema
MODEL_PARAMETER_METASCHEMA = SCHEMA_PATH / "model_parameter.metaschema.yml"
# Path to model parameter description metaschema
MODEL_PARAMETER_DESCRIPTION_METASCHEMA = (
    SCHEMA_PATH / "model_parameter_and_data_schema.metaschema.yml"
)
# Path to sim_telarray meta-parameter metaschema
SIM_TELARRAY_META_PARAMETER_METASCHEMA = SCHEMA_PATH / "sim_telarray_meta_parameter.metaschema.yml"
# Path to sim_telarray meta-parameter registry
SIM_TELARRAY_META_PARAMETER_REGISTRY = SCHEMA_PATH / "sim_telarray_meta_parameters.schema.yml"
# Path to model parameter schema files
MODEL_PARAMETER_SCHEMA_PATH = SCHEMA_PATH / "model_parameters"
# URL to model parameter schema files
MODEL_PARAMETER_SCHEMA_URL = SCHEMA_URL + "/model_parameters"
# Path to resource files
RESOURCE_PATH = files("simtools") / "resources"
# Paths to test resources
_DEFAULT_TEST_RESOURCES_ROOT = Path("tests/unit_tests/resources")


def _configured_test_resources_root():
    """Return the test-resource root configured through environment variables."""
    tests_tag = _configured_test_resources_tag()
    configured_path = os.environ.get("SIMTOOLS_TEST_RESOURCES")
    if configured_path:
        return Path(configured_path).expanduser()

    tests_path = os.environ.get("SIMTOOLS_TESTS_PATH")
    if tests_path:
        tests_tag = tests_tag or _default_test_resources_tag()
        if tests_tag:
            return Path(tests_path).expanduser() / tests_tag / "integration_tests"

    return None


def _configured_test_resources_tag():
    """Return the configured test-resource tag after validating legacy settings."""
    tests_tag = os.environ.get("SIMTOOLS_TESTS_TAG")
    legacy_tag = os.environ.get("SIMTOOLS_TESTS_VERSION")
    if tests_tag and legacy_tag and tests_tag != legacy_tag:
        raise ValueError(
            "SIMTOOLS_TESTS_TAG and SIMTOOLS_TESTS_VERSION must match when both are set."
        )
    return tests_tag or legacy_tag


def _default_test_resources_tag():
    """Return the catalog default tag for the versioned test resources."""
    # Import lazily to keep constants independent of the dependency catalog during module
    # initialization. This also lets installed applications use the same default as pytest.
    from simtools import dependency_versions  # pylint: disable=import-outside-toplevel

    try:
        test_resources = dependency_versions.load_dependency_catalog()["simtools-tests"]
    except FileNotFoundError, KeyError:
        return None
    return test_resources.get("tag", test_resources.get("version"))


def get_test_resources_root():
    """Return the active test-resource root."""
    return _configured_test_resources_root() or TEST_RESOURCES_ROOT


TEST_RESOURCES_ROOT = _configured_test_resources_root() or _DEFAULT_TEST_RESOURCES_ROOT
TEST_RESOURCES_STATIC = str(TEST_RESOURCES_ROOT / "static")
TEST_RESOURCES_GENERATED = str(TEST_RESOURCES_ROOT / "generated")
TEST_RESOURCES_DOWNLOADED = str(TEST_RESOURCES_ROOT / "downloaded")

# Maximum value allowed for random seeds in sim_telarray
SIMTEL_MAX_SEED = 2147483647
# Maximum include filename length accepted by sim_telarray parser (80-char getword buffer).
# The include token is written as "<filename>", so keep the filename itself safely below 80 chars.
SIM_TELARRAY_INCLUDE_FILENAME_MAX_LENGTH = 77
# Maximum value allowed for random seeds in CORSIKA
CORSIKA_MAX_SEED = 900000000

# Default repository URLs for simulations and computing resources
DEFAULT_SIMULATIONS_REPO = "https://gitlab.cta-observatory.org/cta-science/simulations"
DEFAULT_COMPUTING_REPO = "https://gitlab.cta-observatory.org/cta-computing"
DEFAULT_SIMULATION_MODELS = f"{DEFAULT_SIMULATIONS_REPO}/simulation-model/simulation-models.git"
DEFAULT_SIMULATION_WORKFLOWS = (
    f"{DEFAULT_SIMULATIONS_REPO}/simulation-model/simulation-model-parameter-setting.git"
)
