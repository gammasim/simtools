"""Project wide constants."""

from importlib.resources import files

# Schema path
SCHEMA_PATH = files("simtools") / "schemas"
# Path to metadata jsonschema
METADATA_JSON_SCHEMA = SCHEMA_PATH / "metadata.metaschema.yml"
# Path to plotting configuration json schema
PLOT_CONFIG_SCHEMA = SCHEMA_PATH / "plot_configuration.metaschema.yml"
# Path to model parameter metaschema
MODEL_PARAMETER_METASCHEMA = SCHEMA_PATH / "model_parameter.metaschema.yml"
# Path to model parameter description metaschema
MODEL_PARAMETER_DESCRIPTION_METASCHEMA = (
    SCHEMA_PATH / "model_parameter_and_data_schema.metaschema.yml"
)
# Path to model parameter schema files
MODEL_PARAMETER_SCHEMA_PATH = SCHEMA_PATH / "model_parameters"
# URL to model parameter schema files
MODEL_PARAMETER_SCHEMA_URL = (
    "https://raw.githubusercontent.com/gammasim/simtools/main/src/simtools/schemas/"
    "/model_parameters"
)
# Path to resource files
RESOURCE_PATH = files("simtools") / "resources"

# Maximum value allowed for random seeds in sim_telarray
SIMTEL_MAX_SEED = 2147483647
# Maximum value allowed for random seeds in CORSIKA
CORSIKA_MAX_SEED = 900000000

# Default repository URLs for simulations and computing resources
DEFAULT_SIMULATIONS_REPO = "https://gitlab.cta-observatory.org/cta-science/simulations"
DEFAULT_COMPUTING_REPO = "https://gitlab.cta-observatory.org/cta-computing"
DEFAULT_SIMULATION_MODELS = f"{DEFAULT_SIMULATIONS_REPO}/simulation-model/simulation-models.git"
DEFAULT_SIMULATION_WORKFLOWS = (
    f"{DEFAULT_SIMULATIONS_REPO}/simulation-model/simulation-model-parameter-setting.git"
)
