"""Project wide constants."""

from importlib.resources import files

# Path to metadata jsonschema
METADATA_JSON_SCHEMA = files("simtools") / "schemas/metadata.metaschema.yml"

# Path to model parameter schema files
MODEL_PARAMETER_SCHEMA_PATH = files("simtools") / "schemas/model_parameters"
