"""Read and combine production-table files from a model repository."""

import logging
from pathlib import Path

from packaging.version import InvalidVersion, Version

from simtools.utils import names

logger = logging.getLogger(__name__)


def collect_production_descriptions(info_files, read_info):
    """Collect sorted production descriptions using the supplied document reader."""

    def sort_key(path):
        try:
            return (0, Version(path.parent.name))
        except InvalidVersion:
            return (1, path.parent.name)

    descriptions = []
    for path in sorted(info_files, key=sort_key):
        info = read_info(path)
        description = str(info.get("description", "")).replace("\n", " ").strip()
        descriptions.append((str(info.get("model_version", path.parent.name)), description))
    return descriptions


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
