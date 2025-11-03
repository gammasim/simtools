"""Name utilities for array elements, sites, and model parameters.

Naming in simtools:

* 'site': South or North
* 'array element': e.g., LSTN-01, MSTN-01, ...
* 'array element type': e.g., LSTN, MSTN, ...
* 'array element ID': e.g., 01, 02, ...
* 'array element design type': e.g., design, test
* 'instrument class key': e.g., telescope, camera, structure
* 'db collection': e.g., telescopes, sites, calibration_devices

"""

import json
import logging
import re
from functools import cache
from pathlib import Path

import yaml

from simtools.constants import (
    MODEL_PARAMETER_DESCRIPTION_METASCHEMA,
    MODEL_PARAMETER_SCHEMA_PATH,
    RESOURCE_PATH,
)

_logger = logging.getLogger(__name__)

# Mapping of db collection names to class keys
db_collections_to_class_keys = {
    "sites": ["Site"],
    "telescopes": ["Structure", "Camera", "Telescope"],
    "calibration_devices": ["Calibration"],
    "configuration_sim_telarray": ["configuration_sim_telarray"],
    "configuration_corsika": ["configuration_corsika"],
}


@cache
def array_elements():
    """
    Get array elements and their properties.

    Returns
    -------
    dict
        Array elements.
    """
    # for efficiency reason, no functions from simtools.utils.general are used here
    with open(Path(RESOURCE_PATH) / "array_elements.yml", encoding="utf-8") as file:
        return yaml.safe_load(file)["data"]


@cache
def array_element_common_identifiers():
    """
    Get array element IDs from CTAO common identifier.

    Returns
    -------
    dict, dict
        Dictionary mapping array element names to their IDs and vice versa.
    """
    # for efficiency reason, no functions from simtools.utils.general are used here
    id_to_name = {}
    with open(Path(RESOURCE_PATH) / "array-element-ids.json", encoding="utf-8") as file:
        data = json.load(file)
    id_to_name = {e["id"]: e["name"] for e in data["array_elements"]}
    name_to_id = {e["name"]: e["id"] for e in data["array_elements"]}
    return id_to_name, name_to_id


@cache
def simulation_software():
    """
    Get simulation software names from the meta schema definition.

    Returns
    -------
    list
        List of simulation software names.
    """
    with open(Path(MODEL_PARAMETER_DESCRIPTION_METASCHEMA), encoding="utf-8") as file:
        schema = yaml.safe_load(file)
        return schema["definitions"]["SimulationSoftwareName"]["enum"]


@cache
def site_names():
    """
    Get site names.

    The list of sites is derived from the sites listed in array element definition file.
    Return a dictionary for compatibility with the validation '_validate_name' routine.

    Returns
    -------
    dict
        Site names.
    """
    return {
        site: [site.lower()]
        for entry in array_elements().values()
        for site in (entry["site"] if isinstance(entry["site"], list) else [entry["site"]])
    }


@cache
def array_element_design_types(array_element_type):
    """
    Get array element design type (e.g., 'design' or 'flashcam').

    Default values are ['design', 'test'].

    Parameters
    ----------
    array_element_type
        Array element type

    Returns
    -------
    list
        Array element design types.
    """
    default_types = ["design", "test"]
    if array_element_type is None:
        return default_types
    try:
        return array_elements()[array_element_type].get("design_types", default_types)
    except KeyError as exc:
        raise ValueError(f"Invalid name {array_element_type}") from exc


def is_design_type(array_element_name):
    """
    Check if array element is a design type (e.g., "MSTS-FlashCam" or "LSTN-design").

    Parameters
    ----------
    array_element_name: str
        Array element name.

    Returns
    -------
    bool
        True if array element is a design type.
    """
    return get_array_element_id_from_name(array_element_name) in array_element_design_types(
        get_array_element_type_from_name(array_element_name)
    )


@cache
def _load_model_parameters():
    """
    Get model parameters properties from schema files.

    For schema files including multiple schemas, only the first one is returned
    (as this is the most recent definition).

    Returns
    -------
    dict
        Model parameters definitions for all model parameters.
    """
    _parameters = {}
    for schema_file in Path(MODEL_PARAMETER_SCHEMA_PATH).rglob("*.yml"):
        with open(schema_file, encoding="utf-8") as f:
            data = next(yaml.safe_load_all(f))
            _parameters[data["name"]] = data
    return _parameters


def model_parameters(class_key_list=None):
    """
    Get model parameters and their properties for a given instrument class key.

    Returns all model parameters if class_key is None.

    Parameters
    ----------
    class_key: str, None
        Class key (e.g., "telescope", "camera", structure").

    Returns
    -------
    dict
        Model parameters definitions.
    """
    _parameters = {}
    if class_key_list is None:
        return _load_model_parameters()
    for key, value in _load_model_parameters().items():
        if value.get("instrument", {}).get("class", "") in class_key_list:
            _parameters[key] = value
    return _parameters


def site_parameters():
    """Return site model parameters."""
    return model_parameters(class_key_list=tuple(db_collections_to_class_keys["sites"]))


def telescope_parameters():
    """Return telescope model parameters."""
    return model_parameters(class_key_list=tuple(db_collections_to_class_keys["telescopes"]))


def instrument_class_key_to_db_collection(class_name):
    """Convert instrument class key to collection name."""
    for collection, classes in db_collections_to_class_keys.items():
        if class_name in classes:
            return collection
    raise ValueError(f"Class {class_name} not found")


def db_collection_to_instrument_class_key(collection_name="telescopes"):
    """Return list of instrument classes for a given collection."""
    try:
        return db_collections_to_class_keys[collection_name]
    except KeyError as exc:
        raise KeyError(f"Invalid collection name {collection_name}") from exc


def validate_array_element_id_name(array_element_id, array_element_type=None):
    """
    Validate array element ID.

    Allowed IDs are
    - design types (for design array elements or testing)
    - array element ID (e.g., 1, 5, 15)
    - test (for testing)

    Parameters
    ----------
    name: str or int
        Array element ID name.
    array_element_type: str
        Array element type (e.g., LSTN, MSTN).

    Returns
    -------
    str
        Validated array element ID (added leading zeros, e.g., 1 is converted to 01).

    Raises
    ------
    ValueError
        If name is not valid.
    """
    if isinstance(array_element_id, int) or array_element_id.isdigit():
        return f"{int(array_element_id):02d}"
    if array_element_id in array_element_design_types(array_element_type):
        return str(array_element_id)
    raise ValueError(f"Invalid array element ID name {array_element_id}")


def validate_site_name(site_name):
    """
    Validate site name.

    Parameters
    ----------
    site_name: str
        Site name.

    Returns
    -------
    str
        Validated name.
    """
    return _validate_name(site_name, site_names())


def _validate_name(name, all_names):
    """
    Validate name given the all_names options.

    For each key in all_names, a list of options is given.
    If name is in this list, the key name is returned.

    Parameters
    ----------
    name: str
        Name to validate.
    all_names: dict
        Dictionary with valid names.

    Returns
    -------
    str
        Validated name.

    Raises
    ------
    ValueError
        If name is not valid.
    """
    for key in all_names.keys():
        if isinstance(all_names[key], list) and name.lower() in [
            item.lower() for item in all_names[key]
        ]:
            return key
        if name.lower() == key.lower():
            return key

    msg = f"Invalid name {name}"
    raise ValueError(msg)


def validate_array_element_type(array_element_type):
    """
    Validate array element type (e.g., LSTN, MSTN).

    Parameters
    ----------
    array_element_type: str
        Array element type.

    Returns
    -------
    str
        Validated name.
    """
    return _validate_name(array_element_type, array_elements())


def validate_array_element_name(array_element_name):
    """
    Validate array element name (e.g., MSTx-NectarCam, MSTN-01).

    Forgiving validation, is it allows also to give a site name (e.g., OBS-North).

    Parameters
    ----------
    array_element_name: str
        Array element name.

    Returns
    -------
    str
        Validated name.
    """
    try:
        _array_element_type, _array_element_id = array_element_name.split("-")
    except ValueError as exc:
        msg = f"Invalid name {array_element_name}"
        raise ValueError(msg) from exc
    if _array_element_type == "OBS":
        return validate_site_name(_array_element_id)
    return (
        _validate_name(_array_element_type, array_elements())
        + "-"
        + validate_array_element_id_name(_array_element_id, _array_element_type)
    )


def generate_array_element_name_from_type_site_id(array_element_type, site, array_element_id):
    """
    Generate a new array element name from array element type, site, and array element ID.

    Parameters
    ----------
    array_element_type: str
        Array element type.
    site: str
        Site name.
    array_element_id: str
        Array element ID.

    Returns
    -------
    str
        Array element name.
    """
    _short_site = validate_site_name(site)[0]
    _val_id = validate_array_element_id_name(array_element_id, array_element_type)
    return f"{array_element_type}{_short_site}-{_val_id}"


def get_array_element_type_from_name(array_element_name):
    """
    Get array element type from array element name (e.g "MSTN" from "MSTN-01").

    For sites, return site name.

    Parameters
    ----------
    array_element_name: str
        Array element name

    Returns
    -------
    str
        Array element type.
    """
    try:  # e.g. instrument is 'North' as given for the site parameters
        return validate_site_name(array_element_name)
    except ValueError:  # any other telescope or calibration device
        return _validate_name(array_element_name.split("-")[0], array_elements())


def get_array_element_id_from_name(array_element_name):
    """
    Get array element ID from array element name, (e.g. "01" from "MSTN-01").

    Parameters
    ----------
    array_element_name: str
        Array element name

    Returns
    -------
    str
        Array element ID.
    """
    try:
        return validate_array_element_id_name(
            array_element_name.split("-")[1], array_element_name.split("-")[0]
        )
    except IndexError as exc:
        raise ValueError(f"Invalid name {array_element_name}") from exc


def get_common_identifier_from_array_element_name(array_element_name, default_return=None):
    """
    Get numerical common identifier from array element name as used by CTAO.

    Common identifiers are numerical IDs used by the CTAO ACADA and DPPS systems.

    Parameters
    ----------
    array_element_name: str
        Array element name (e.g. LSTN-01)

    Returns
    -------
    int
        Common identifier.
    """
    _, name_to_id = array_element_common_identifiers()
    try:
        return name_to_id[array_element_name]
    except KeyError as exc:
        if default_return is not None:
            return default_return
        raise ValueError(f"Unknown array element name {array_element_name}") from exc


def get_array_element_name_from_common_identifier(common_identifier):
    """
    Get array element name from common identifier as used by CTAO.

    Common identifiers are numerical IDs used by the CTAO ACADA and DPPS systems.

    Parameters
    ----------
    common_identifier: int
        Common identifier.

    Returns
    -------
    str
        Array element name.
    """
    id_to_name, _ = array_element_common_identifiers()
    try:
        return id_to_name[common_identifier]
    except KeyError as exc:
        raise ValueError(f"Unknown common identifier {common_identifier}") from exc


def get_list_of_array_element_types(
    array_element_class="telescopes", site=None, observatory="CTAO"
):
    """
    Get list of array element types (e.g., ["LSTN", "MSTN"] for the Northern site).

    Parameters
    ----------
    array_element_class: str
        Array element class
    site: str
        Site name (e.g., South or North).

    Returns
    -------
    list
        List of array element types.
    """
    return sorted(
        [
            key
            for key, value in array_elements().items()
            if value["collection"] == array_element_class
            and (site is None or value["site"] == site)
            and (observatory is None or value["observatory"] == observatory)
        ]
    )


def get_site_from_array_element_name(array_element_name):
    """
    Get site name from array element name (e.g., "South" from "MSTS-01").

    Parameters
    ----------
    array_element_name: str
        Array element name.

    Returns
    -------
    str, list
        Site name(s).
    """
    try:  # e.g. instrument is 'North' as given for the site parameters
        if array_element_name.startswith("OBS"):
            return validate_site_name(array_element_name.split("-")[1])
        return validate_site_name(array_element_name)
    except ValueError:  # e.g. instrument is 'LSTN' as given for the array element types
        return array_elements()[get_array_element_type_from_name(array_element_name)]["site"]


def get_collection_name_from_array_element_name(array_element_name, array_elements_only=True):
    """
    Get collection name (e.g., telescopes, calibration_devices) of an array element from its name.

    Parameters
    ----------
    array_element_name: str
        Array element name (e.g. LSTN-01)
    array_elements_only: bool
        If True, only array elements are considered (e.g. "OBS-North" will raise a ValueError).

    Returns
    -------
    str
        Collection name .

    Raises
    ------
    ValueError
        If name is not a valid array element name.
    """
    try:
        return array_elements()[get_array_element_type_from_name(array_element_name)]["collection"]
    except (ValueError, KeyError) as exc:
        if array_elements_only:
            raise ValueError(f"Invalid array element name {array_element_name}") from exc
    try:
        if array_element_name.startswith("OBS") or validate_site_name(array_element_name):
            return "sites"
    except ValueError:
        pass
    if array_element_name in {
        "configuration_sim_telarray",
        "configuration_corsika",
        "Files",
        "Dummy-Telescope",
    }:
        return array_element_name
    raise ValueError(f"Invalid array element name {array_element_name}")


def get_collection_name_from_parameter_name(parameter_name):
    """
    Get the db collection name for a given parameter.

    Parameters
    ----------
    parameter_name: str
        Name of the parameter.

    Returns
    -------
    str
        Collection name.

    Raises
    ------
    KeyError
        If the parameter name is not found in the list of model parameters
    """
    _parameter_names = model_parameters()
    try:
        class_key = _parameter_names[parameter_name].get("instrument", {}).get("class")
    except KeyError as exc:
        raise KeyError(f"Parameter {parameter_name} without schema definition") from exc
    return instrument_class_key_to_db_collection(class_key)


def get_simulation_software_name_from_parameter_name(
    parameter_name,
    software_name="sim_telarray",
    set_meta_parameter=False,
):
    """
    Get the name used in the given simulation software from the model parameter name.

    Name convention is expected to be defined in the model parameter schema.
    Returns the parameter name if no simulation software name is found.

    Parameters
    ----------
    parameter_name: str
        Model parameter name.
    simulation_software: str
        Simulation software name.
    set_meta_parameter: bool
        If True, return values with 'set_meta_parameter' field set to True.

    Returns
    -------
    str
        Simtel parameter name.
    """
    _parameter = model_parameters().get(parameter_name)
    if not _parameter:
        raise KeyError(f"Parameter {parameter_name} without schema definition")

    for software in _parameter.get("simulation_software", []):
        if (
            software.get("name") == software_name
            and software.get("set_meta_parameter", False) is set_meta_parameter
        ):
            return software.get("internal_parameter_name", parameter_name)

    return None


def simtel_config_file_name(
    site,
    model_version,
    array_name=None,
    telescope_model_name=None,
    label=None,
    extra_label=None,
):
    """
    sim_telarray config file name for a telescope.

    Parameters
    ----------
    site: str
        South or North.
    telescope_model_name: str
        LST-1, MST-FlashCam, ...
    model_version: str
        Version of the model.
    label: str
        Instance label.
    extra_label: str
        Extra label in case of multiple telescope config files.

    Returns
    -------
    str
        File name.
    """
    name = "CTA"
    name += f"-{array_name}" if array_name is not None else ""
    name += f"-{site}"
    name += f"-{telescope_model_name}" if telescope_model_name is not None else ""
    name += f"-{model_version}"
    name += f"_{label}" if label is not None else ""
    name += f"_{extra_label}" if extra_label is not None else ""
    name += ".cfg"
    return name


def simtel_single_mirror_list_file_name(
    site, telescope_model_name, model_version, mirror_number, label
):
    """
    sim_telarray mirror list file with a single mirror.

    Parameters
    ----------
    site: str
        South or North.
    telescope_model_name: str
        North-LST-1, South-MST-FlashCam, ...
    model_version: str
        Version of the model.
    mirror_number: int
        Mirror number.
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    name = f"CTA-single-mirror-list-{site}-{telescope_model_name}-{model_version}"
    name += f"-mirror{mirror_number}"
    name += f"_{label}" if label is not None else ""
    name += ".dat"
    return name


def generate_file_name(
    file_type,
    suffix,
    site,
    telescope_model_name,
    zenith_angle,
    azimuth_angle=None,
    off_axis_angle=None,
    source_distance=None,
    mirror_number=None,
    label=None,
    extra_label=None,
):
    """
    Generate a file name for output, config, or plotting.

    Used e.g., to generate camera_efficiency and ray_tracing output files.

    Parameters
    ----------
    file_type: str
        Type of file (e.g., config, output, plot)
    suffix: str
        File suffix
    site: str
        South or North.
    telescope_model_name: str
        LSTN-01, MSTS-01, ...
    zenith_angle: float
        Zenith angle (deg).
    azimuth_angle: float
        Azimuth angle (deg).
    off_axis_angle: float
        Off-axis angle (deg).
    source_distance: float
        Source distance (km).
    mirror_number: int
        Mirror number.
    label: str
        Instance label.
    extra_label: str
        Extra label.

    Returns
    -------
    str
        File name.
    """
    name = f"{file_type}_{site}_{telescope_model_name}"
    name += f"_d{source_distance:.1f}km" if source_distance is not None else ""
    name += f"_za{float(zenith_angle):.1f}deg"
    name += f"_off{off_axis_angle:.3f}deg" if off_axis_angle is not None else ""
    name += f"_azm{round(azimuth_angle):03}deg" if azimuth_angle is not None else ""
    name += f"_mirror{mirror_number}" if mirror_number is not None else ""
    name += f"_{label}" if label is not None else ""
    name += f"_{extra_label}" if extra_label is not None else ""
    name += f"{suffix}"
    return name


def sanitize_name(name):
    """
    Sanitize name to be a valid Python identifier.

    - Replaces spaces with underscores
    - Converts to lowercase
    - Removes characters that are not alphanumerics or underscores
    - If the name starts with a number, prepend an underscore

    Parameters
    ----------
    name: str
        name to be sanitized.

    Returns
    -------
    str:
        Sanitized name.

    Raises
    ------
    ValueError:
        if the string name can not be sanitized.
    """
    if name is None:
        return None
    sanitized = name.lower()
    sanitized = sanitized.replace(" ", "_")
    # Remove characters that are not alphanumerics or underscores
    sanitized = re.sub(r"\W|^(?=\d)", "_", sanitized)
    if not sanitized.isidentifier():
        msg = f"The string {name} could not be sanitized."
        _logger.error(msg)
        raise ValueError(msg)
    return sanitized


def file_name_with_version(file_name, suffix):
    """
    Return a file name including a semantic version with the correct suffix.

    Replaces 'Path.suffix()', which removes trailing numbers (and therefore version numbers).

    Parameters
    ----------
    file_name: str
        File name.
    suffix: str
        File suffix.

    Returns
    -------
    Path
        File name with version number.
    """
    if file_name is None or suffix is None:
        return None
    file_name = str(file_name)
    if re.search(r"\d{1,8}\.\d{1,8}\.\d{1,8}\Z", file_name):
        return Path(file_name + suffix)
    return Path(file_name).with_suffix(suffix)
