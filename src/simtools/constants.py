"""Project wide constants."""

from pathlib import Path

# Path to metadata jsonschema
METADATA_JSON_SCHEMA = Path(__file__).parent / "../simtools/schemas/metadata.metaschema.yml"

# Path to model parameter schema files
MODEL_PARAMETER_SCHEMA_PATH = Path(__file__).parent / "../simtools/schemas/model_parameters"
