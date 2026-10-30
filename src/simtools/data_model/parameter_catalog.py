"""Access model parameter schemas and their catalog metadata."""

from functools import cache

from simtools.constants import MODEL_PARAMETER_DESCRIPTION_METASCHEMA, MODEL_PARAMETER_SCHEMA_PATH
from simtools.data_model import schema_loader
from simtools.utils import names

# Mapping of model repository collection names to instrument class keys.
collections_to_class_keys = {
    "sites": ["Site"],
    "telescopes": ["Structure", "Camera", "Telescope"],
    "calibration_devices": ["Calibration"],
    "configuration_sim_telarray": ["configuration_sim_telarray"],
    "configuration_corsika": ["configuration_corsika"],
}

SIM_TELARRAY_GLOBAL_PARAMETERS = frozenset(
    {"iobuf_maximum", "iobuf_output_maximum", "random_generator", "save_pe_with_amplitude"}
)


def is_global_sim_telarray_parameter(parameter_name):
    """Return whether a sim_telarray parameter has global scope."""
    return parameter_name in SIM_TELARRAY_GLOBAL_PARAMETERS


def get_model_parameter_scope(collection_name, instrument, parameter_name):
    """Return the instrument scope used to store a model parameter."""
    if collection_name == "configuration_corsika" or (
        collection_name == "configuration_sim_telarray"
        and is_global_sim_telarray_parameter(parameter_name)
    ):
        return "global"
    return "global" if instrument == "global" else instrument


def get_sim_telarray_parameter_scopes(
    array_element_name, design_model=None, ignore_missing_design_model=False
):
    """Return the global, design, and telescope scopes for sim_telarray."""
    if array_element_name in (None, "global"):
        return ["global"]
    if names.is_design_type(array_element_name):
        return ["global", array_element_name]
    if design_model:
        return ["global", design_model, array_element_name]
    if ignore_missing_design_model:
        element_type = names.get_array_element_type_from_name(array_element_name)
        return ["global", array_element_name, f"{element_type}-01", f"{element_type}-design"]
    raise KeyError(f"Missing design model for {array_element_name}")


@cache
def simulation_software():
    """Get simulation software names from the meta schema definition."""
    schema = schema_loader.load_schema(MODEL_PARAMETER_DESCRIPTION_METASCHEMA, "latest")
    return schema["definitions"]["SimulationSoftwareName"]["enum"]


@cache
def _load_model_parameters():
    """Load the latest definition of each model parameter schema."""
    parameters, schema_files = schema_loader.get_model_parameter_schema_files(
        MODEL_PARAMETER_SCHEMA_PATH
    )
    return {
        parameter: schema_loader.load_schema(schema_file, "latest")
        for parameter, schema_file in zip(parameters, schema_files)
    }


def model_parameters(class_key_list=None):
    """Get model parameter definitions, optionally filtered by class key."""
    if class_key_list is None:
        return _load_model_parameters()
    return {
        key: value
        for key, value in _load_model_parameters().items()
        if value.get("instrument", {}).get("class", "") in class_key_list
    }


def site_parameters():
    """Return site model parameters."""
    return model_parameters(class_key_list=tuple(collections_to_class_keys["sites"]))


def instrument_class_key_to_collection(class_name):
    """Convert instrument class key to collection name."""
    for collection, classes in collections_to_class_keys.items():
        if class_name in classes:
            return collection
    raise ValueError(f"Class {class_name} not found")


@cache
def _model_parameter_for_collection(parameter_name):
    """Return one parameter schema without loading unrelated parameter schemas."""
    schema_file = MODEL_PARAMETER_SCHEMA_PATH / f"{parameter_name}.schema.yml"
    try:
        parameter = schema_loader.load_schema(schema_file, "latest")
    except FileNotFoundError:
        parameter = None
    if parameter is None or parameter.get("name") != parameter_name:
        parameter = model_parameters().get(parameter_name)
    if parameter is None:
        raise KeyError(f"Parameter {parameter_name} without schema definition")
    return parameter


def collection_to_instrument_class_key(collection_name="telescopes"):
    """Return instrument classes for a model repository collection."""
    try:
        return collections_to_class_keys[collection_name]
    except KeyError as exc:
        raise KeyError(f"Invalid collection name {collection_name}") from exc


def get_collection_name_from_parameter_name(parameter_name):
    """Get the model repository collection for a model parameter."""
    parameter = _model_parameter_for_collection(parameter_name)
    class_key = parameter.get("instrument", {}).get("class")
    return instrument_class_key_to_collection(class_key)


def get_simulation_software_name_from_parameter_name(parameter_name, software_name="sim_telarray"):
    """Get the parameter name used by a simulation software, if mapped."""
    parameter = model_parameters().get(parameter_name)
    if not parameter:
        raise KeyError(f"Parameter {parameter_name} without schema definition")
    for software in parameter.get("simulation_software", []):
        if software.get("name") == software_name:
            return software.get("internal_parameter_name", parameter_name)
    return None


def get_simulation_software_meta_parameter_mode(parameter_name, software_name="sim_telarray"):
    """Return the metadata serialization mode for a software mapping."""
    parameter = model_parameters().get(parameter_name)
    if not parameter:
        raise KeyError(f"Parameter {parameter_name} without schema definition")
    for software in parameter.get("simulation_software", []):
        if software.get("name") == software_name:
            return software.get("meta_parameter_mode")
    return None
