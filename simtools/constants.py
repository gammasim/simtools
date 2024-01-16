"""Project wide constants."""

# Path to metadata jsonschema
METADATA_JSON_SCHEMA = "schemas/metadata.metaschema.yml"
# Path to data and modelparameter jsonschema
DATA_JSON_SCHEMA = "schemas/data.metaschema.yml"

# URL to the schema repository
SCHEMA_URL = "https://raw.githubusercontent.com/gammasim/workflows/main/schemas/"

# URL to the repository with the simulation model
# (temporary used for the development of the simulation model schema;
# will be replaced by the database
# TODO Temporarily set to a local path for easier development
# SIMULATION_MODEL_URL = "../simulation_model/verified_model/"
SIMULATION_MODEL_URL = None
