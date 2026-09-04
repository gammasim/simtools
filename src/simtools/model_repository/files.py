"""Read and combine production-table files from a model repository."""

import logging
from pathlib import Path

from packaging.version import Version

from simtools.io import ascii_handler
from simtools.utils import names

logger = logging.getLogger(__name__)


def get_production_table_files(model_path):
    """Return production JSON files and the model version they belong to."""
    models = [model_path.name]
    info_path = model_path / "info.yml"
    if info_path.exists():
        info = ascii_handler.collect_data_from_file(file_name=info_path)
        if info.get("model_update") == "patch_update":
            models.extend(info.get("model_version_history", []))
    models = sorted(set(models), key=Version)
    return [
        (model, file)
        for model in models
        for file in sorted((model_path.parent / model).rglob("*json"))
    ]


def read_production_tables(model_path, collection_name=None, production_files=None):
    """Read and merge production tables for one model version."""
    model_dict = {}
    if production_files is None:
        production_files = get_production_table_files(model_path)
    for model, file in production_files:
        if collection_name and (
            names.get_collection_name_from_array_element_name(file.stem, False) != collection_name
        ):
            continue
        _read_production_table(model_dict, file, model)

    for table in model_dict.values():
        table["model_version"] = model_path.name
    _remove_deprecated_model_parameters(model_dict)
    return model_dict


def read_production_tables_from_documents(model_version, documents, collection_name=None):
    """Read and merge production tables from parsed repository documents.

    ``documents`` contains ``(model_name, file_name, document)`` tuples.  The
    representation is shared by filesystem and Git readers so that both
    sources apply patch history and production-table merging identically.
    """
    model_dict = {}
    for model_name, file_name, document in documents:
        if collection_name and (
            names.get_collection_name_from_array_element_name(Path(file_name).stem, False)
            != collection_name
        ):
            continue
        _merge_production_table(model_dict, Path(file_name).stem, model_name, document)

    for table in model_dict.values():
        table["model_version"] = model_version
    _remove_deprecated_model_parameters(model_dict)
    return model_dict


def _read_production_table(model_dict, file, model_name):
    """Read one production-table JSON file into an aggregate."""
    parameter_dict = ascii_handler.collect_data_from_file(file_name=file)
    _merge_production_table(model_dict, file.stem, model_name, parameter_dict)


def _merge_production_table(model_dict, array_element, model_name, parameter_dict):
    """Merge one parsed production-table document into an aggregate."""
    collection = names.get_collection_name_from_array_element_name(array_element, False)
    model_dict.setdefault(
        collection,
        {
            "collection": collection,
            "model_version": model_name,
            "parameters": {},
            "design_model": {},
            "deprecated_parameters": [],
        },
    )
    if array_element in ("configuration_corsika", "configuration_sim_telarray"):
        model_dict[collection]["parameters"] = parameter_dict["parameters"]
    else:
        model_dict[collection]["parameters"].setdefault(array_element, {}).update(
            parameter_dict["parameters"][array_element]
        )
    try:
        model_dict[collection]["design_model"][array_element] = parameter_dict["design_model"][
            array_element
        ]
    except KeyError:
        pass
    if "deprecated_parameters" in parameter_dict:
        model_dict[collection]["deprecated_parameters"] = parameter_dict["deprecated_parameters"]


def _remove_deprecated_model_parameters(model_dict):
    """Remove deprecated parameters from aggregated production tables."""
    for table in model_dict.values():
        for params in table.get("parameters", {}).values():
            for parameter in table.get("deprecated_parameters", []):
                if parameter in params:
                    logger.info("Removing deprecated parameter %s", parameter)
                    params.pop(parameter)
