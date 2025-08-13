"""Retrieve, merge, and write layout dictionaries."""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.data_model import data_reader
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.data_model.model_data_writer import ModelDataWriter
from simtools.io import ascii_handler, io_handler
from simtools.model.array_model import ArrayModel
from simtools.model.site_model import SiteModel
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
        array_element_ids = ascii_handler.collect_data_from_http(
            url=f"{repository_url}/{branch_name}/array-element-ids.json"
        )
        sub_arrays = ascii_handler.collect_data_from_http(
            url=f"{repository_url}/{branch_name}/subarray-ids.json"
        )
    else:
        array_element_ids = ascii_handler.collect_data_from_file(
            Path(repository_url) / "array-element-ids.json"
        )
        sub_arrays = ascii_handler.collect_data_from_file(
            Path(repository_url) / "subarray-ids.json"
        )

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


def get_array_layouts_from_parameter_file(
    file_path, model_version, db_config, coordinate_system="ground"
):
    """
    Retrieve array layouts from parameter file.

    Parameters
    ----------
    file_path : str or Path
        Path to the array layout parameter file.
    model_version : str
        Model version to retrieve.
    db_config : dict
        Database configuration.
    coordinate_system : str
        Coordinate system to use for the array elements (default is "ground").

    Returns
    -------
    list
        List of dictionaries containing array layout names and their elements.
    """
    array_layouts = ascii_handler.collect_data_from_file(file_path)
    try:
        value = array_layouts["value"]
    except KeyError as exc:
        raise ValueError("Missing 'value' key in layout file.") from exc
    site = array_layouts.get("site")

    layouts = []
    for layout in value:
        layouts.append(
            _get_array_layout_dict(
                db_config,
                model_version,
                site,
                layout.get("elements"),
                layout["name"],
                coordinate_system,
            )
        )
    return layouts


def get_array_layouts_from_db(
    layout_name, site, model_version, db_config, coordinate_system="ground"
):
    """
    Retrieve all array layouts from the database and return as list of astropy tables.

    Parameters
    ----------
    layout_name : str
        Name of the array layout to retrieve (for None, all layouts are retrieved).
    site : str
        Site identifier.
    model_version : str
        Model version to retrieve.
    db_config : dict
        Database configuration.
    coordinate_system : str
        Coordinate system to use for the array elements (default is "ground").

    Returns
    -------
    list
        List of dictionaries containing array layout names and their elements.
    """
    # TODO - this function should be replace by enforce list type from utils.general
    # (wait for simulate-calibration-events merge)
    layout_names = []
    if layout_name:
        layout_names.append(
            layout_name[0]
            if isinstance(layout_name, list) and len(layout_name) == 1
            else layout_name
        )
    else:
        site_model = SiteModel(site=site, model_version=model_version, mongo_db_config=db_config)
        layout_names = site_model.get_list_of_array_layouts()

    layouts = []
    for _layout_name in layout_names:
        layouts.append(
            _get_array_layout_dict(
                db_config, model_version, site, None, _layout_name, coordinate_system
            )
        )
    if len(layouts) == 1:
        return layouts[0]
    return layouts


def get_array_layouts_using_telescope_lists_from_db(
    telescope_lists, site, model_version, db_config, coordinate_system="ground"
):
    """
    Retrieve array layouts from the database using telescope lists.

    Parameters
    ----------
    telescope_lists : list
        List of telescope lists to retrieve array layouts for.
    site : str
        Site identifier.
    model_version : str
        Model version to retrieve.
    db_config : dict
        Database configuration.
    coordinate_system : str
        Coordinate system to use for the array elements (default is "ground").

    Returns
    -------
    list
        List of dictionaries containing array layout names and their elements.

    """
    layouts = []
    for telescope_list in telescope_lists:
        _site = site
        if _site is None:
            sites = {names.get_site_from_array_element_name(t) for t in telescope_list}
            if len(sites) != 1:
                raise ValueError(
                    f"Telescope list contains elements from multiple sites: {sites}."
                    "Please specify a site."
                )
            _site = sites.pop()

        layouts.append(
            _get_array_layout_dict(
                db_config, model_version, _site, telescope_list, None, coordinate_system
            )
        )
    return layouts


def get_array_layouts_from_file(file_path):
    """
    Retrieve array layout(s) from astropy table file(s).

    Parameters
    ----------
    file_path : str or Path or list of str or list of Path
        Path(s) to array layout files(s).

    Returns
    -------
    list
        List of dictionaries containing array layout names and their elements.
    """
    if isinstance(file_path, str | Path):
        file_path = [file_path]

    layouts = []
    for _file in file_path:
        layouts.append(
            {
                "name": (Path(_file).name).split(".")[0],
                "array_elements": data_reader.read_table_from_file(file_name=_file),
            }
        )
    return layouts


def _get_array_layout_dict(
    db_config, model_version, site, telescope_list, layout_name, coordinate_system
):
    """Return array layout dictionary for a given telescope list."""
    array_model = ArrayModel(
        mongo_db_config=db_config,
        model_version=model_version,
        site=site,
        array_elements=telescope_list,
        layout_name=layout_name,
    )
    return {
        "name": layout_name if layout_name else "list",
        "site": site,
        "array_elements": array_model.export_array_elements_as_table(
            coordinate_system=coordinate_system
        ),
    }


def get_array_elements_from_db_for_layouts(layouts, site, model_version, db_config):
    """
    Get list of array elements from the database for given list of layout names.

    Structure of the returned dictionary::

        {
            "layout_name_1": [telescope_id_1, telescope_id_2, ...],
            "layout_name_2": [telescope_id_3, telescope_id_4, ...],
            ...
        }

    Parameters
    ----------
    layouts : list[str]
        List of layout names to read. If "all", read all available layouts.
    site : str
        Site name for the array layouts.
    model_version : str
        Model version for the array layouts.
    db_config : dict
        Database configuration dictionary.

    Returns
    -------
    dict
        Dictionary mapping layout names to telescope IDs.
    """
    site_model = SiteModel(site=site, model_version=model_version, mongo_db_config=db_config)
    layout_names = site_model.get_list_of_array_layouts() if layouts == ["all"] else layouts
    layout_dict = {}
    for layout_name in layout_names:
        layout_dict[layout_name] = site_model.get_array_elements_for_layout(layout_name)
    return layout_dict
