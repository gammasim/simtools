"""Retrieve, merge, and write layout dictionaries."""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.data_model.model_data_writer import ModelDataWriter
from simtools.io_operations import io_handler
from simtools.utils import names

_logger = logging.getLogger(__name__)


def retrieve_ctao_array_layouts(site, repository_url, branch_name="main"):
    """
    Retrieve array layouts from CTAO common identifiers repository.

    Parameters
    ----------
    site : str
        Site identifier.
    repository_url : str
        URL or path to CTAO common identifiers
    branch_name : str
        Repository branch to use for CTAO common identifiers.

    Returns
    -------
    dict
        Array layouts for all CTAO sites.
    """
    _logger.info(f"Retrieving array layouts from {repository_url} on branch {branch_name}.")

    if gen.is_url(repository_url):
        array_element_ids = gen.collect_data_from_http(
            url=f"{repository_url}/{branch_name}/array-element-ids.json"
        )
        sub_arrays = gen.collect_data_from_http(
            url=f"{repository_url}/{branch_name}/subarray-ids.json"
        )
    else:
        array_element_ids = gen.collect_data_from_file(
            Path(repository_url) / "array-element-ids.json"
        )
        sub_arrays = gen.collect_data_from_file(Path(repository_url) / "subarray-ids.json")

    return _get_ctao_layouts_per_site(site, sub_arrays, array_element_ids)


def _get_ctao_layouts_per_site(site, sub_arrays, array_element_ids):
    """
    Get array layouts for CTAO sites.

    Parameters
    ----------
    site : str
        Site identifier.
    sub_arrays : dict
        Sub-array definitions.
    array_element_ids : dict
        Array element definitions.

    Returns
    -------
    dict
        Array layouts for CTAO sites.
    """
    layouts_per_site = []

    for array in sub_arrays.get("subarrays", []):
        elements = []
        for ids in array.get("array_element_ids", []):
            element_name = _get_ctao_array_element_name(ids, array_element_ids)
            if names.get_site_from_array_element_name(element_name) != site:
                break
            elements.append(element_name)
        if len(elements) > 0:
            array_layout = {
                "name": array.get("name"),
                "elements": elements,
            }
            layouts_per_site.append(array_layout)

    _logger.info(f"CTAO array layout definition: {layouts_per_site}")
    return layouts_per_site


def _get_ctao_array_element_name(ids, array_element_ids):
    """Return array element name for common identifier."""
    for element in array_element_ids.get("array_elements", []):
        if element.get("id") == ids:
            return element.get("name")
    return None


def merge_array_layouts(layouts_1, layouts_2):
    """
    Compare two array layout dictionaries and merge them.

    Parameters
    ----------
    layouts_1 : dict
        Array layout dictionary 1.
    layouts_2 : dict
        Array layout dictionary 2.

    Returns
    -------
    dict
        Merged array layout dictionary based on layout_1.
    """
    merged_layout = layouts_1
    for layout_2 in layouts_2:
        layout_found = False
        for layout_1 in layouts_1.get("value", {}):
            if sorted(layout_1["elements"]) == sorted(layout_2["elements"]):
                print(
                    f"Equal telescope list: simtools '{layout_1['name']}' "
                    f"and CTAO '{layout_2['name']}'"
                )
                layout_1["name"] = layout_2["name"]
                layout_found = True
        if not layout_found:
            merged_layout["value"].append(
                {
                    "name": layout_2["name"],
                    "elements": layout_2["elements"],
                }
            )
            _logger.info(f"Adding {layout_2['name']} with {layout_2['elements']}")
    return merged_layout


def write_array_layouts(array_layouts, args_dict, db_config):
    """
    Write array layouts as model parameter.

    Parameters
    ----------
    args_dict : dict
        Command line arguments.
    array_layouts : dict
        Array layouts to be written.
    db_config : dict
        Database configuration.
    """
    site = args_dict.get("site") or array_layouts.get("site")
    _logger.info(f"Writing updated array layouts to the database for site {site}.")

    io_handler_instance = io_handler.IOHandler()
    io_handler_instance.set_paths(
        output_path=args_dict["output_path"],
        use_plain_output_path=args_dict["use_plain_output_path"],
    )
    output_file = io_handler_instance.get_output_file(
        f"array-layouts-{args_dict['updated_parameter_version']}.json"
    )

    ModelDataWriter.dump_model_parameter(
        parameter_name="array_layouts",
        value=array_layouts["value"],
        instrument=site,
        parameter_version=args_dict.get("updated_parameter_version"),
        output_file=output_file,
        use_plain_output_path=args_dict["use_plain_output_path"],
        db_config=db_config,
    )
    MetadataCollector.dump(
        args_dict,
        output_file,
        add_activity_name=True,
    )


def validate_array_layouts_with_db(production_table, array_layouts):
    """
    Validate array layouts against the production table in the database.

    Confirm that every telescope defined in the array layouts exist in the
    production table.

    Parameters
    ----------
    production_table : dict
        Production table from the database.
    array_layouts : dict
        Array layouts to be validated.

    Returns
    -------
    dict
        Validated array layouts.
    """
    db_elements = set(production_table.get("parameters", {}).keys())

    invalid_array_elements = [
        e
        for layout in array_layouts.get("value", [])
        for e in layout.get("elements", [])
        if e not in db_elements
    ]

    if invalid_array_elements:
        raise ValueError(f"Invalid array elements found: {invalid_array_elements}. ")

    return array_layouts
