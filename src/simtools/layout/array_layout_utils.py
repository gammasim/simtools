"""Retrieve, merge, and write layout dictionaries."""

import logging
from pathlib import Path

import astropy.units as u
from astropy.table import QTable, Table

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


def write_array_layouts(array_layouts, args_dict):
    """
    Write array layouts as model parameter.

    Parameters
    ----------
    args_dict : dict
        Command line arguments.
    array_layouts : dict
        Array layouts to be written.
    """
    site = args_dict.get("site") or array_layouts.get("site")
    _logger.info(f"Writing updated array layouts to the database for site {site}.")

    io_handler_instance = io_handler.IOHandler()
    io_handler_instance.set_paths(output_path=args_dict["output_path"])
    output_file = io_handler_instance.get_output_file(
        f"array-layouts-{args_dict['updated_parameter_version']}.json"
    )

    ModelDataWriter.dump_model_parameter(
        parameter_name="array_layouts",
        value=array_layouts["value"],
        instrument=f"OBS-{site}",
        parameter_version=args_dict.get("updated_parameter_version"),
        output_file=output_file,
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


def get_array_layouts_from_parameter_file(file_path, model_version, coordinate_system="ground"):
    """
    Retrieve array layouts from parameter file.

    Parameters
    ----------
    file_path : str or Path
        Path to the array layout parameter file.
    model_version : str
        Model version to retrieve.
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

    return [
        _get_array_layout_dict(
            model_version,
            site,
            layout.get("elements"),
            layout["name"],
            coordinate_system,
        )
        for layout in value
    ]


def get_array_layouts_from_db(layout_name, site, model_version, coordinate_system="ground"):
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
    coordinate_system : str
        Coordinate system to use for the array elements (default is "ground").

    Returns
    -------
    list
        List of dictionaries containing array layout names and their elements.
    """
    layout_names = []
    if layout_name:
        layout_names = gen.ensure_iterable(layout_name)
    else:
        site_model = SiteModel(site=site, model_version=model_version)
        layout_names = site_model.get_list_of_array_layouts()

    layouts = [
        _get_array_layout_dict(model_version, site, None, _layout_name, coordinate_system)
        for _layout_name in layout_names
    ]
    if len(layouts) == 1:
        return layouts[0]
    return layouts


def get_array_layouts_using_telescope_lists_from_db(
    telescope_lists, site, model_version, coordinate_system="ground"
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
            _get_array_layout_dict(model_version, _site, telescope_list, None, coordinate_system)
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

    return [
        {
            "name": Path(_file).stem,
            "array_elements": data_reader.read_table_from_file(file_name=_file),
        }
        for _file in file_path
    ]


def _get_array_layout_dict(model_version, site, telescope_list, layout_name, coordinate_system):
    """Return array layout dictionary for a given telescope list."""
    array_model = ArrayModel(
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


def get_array_elements_from_db_for_layouts(layouts, site, model_version):
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

    Returns
    -------
    dict
        Dictionary mapping layout names to telescope IDs.
    """
    site_model = SiteModel(site=site, model_version=model_version)
    layout_names = site_model.get_list_of_array_layouts() if layouts == ["all"] else layouts
    layout_dict = {}
    for layout_name in layout_names:
        layout_dict[layout_name] = site_model.get_array_elements_for_layout(layout_name)
    return layout_dict


def read_layouts(args_dict):
    """
    Read array layouts from the database or parameter file.

    Parameters
    ----------
    args_dict : dict
        Dictionary with command line arguments.

    Returns
    -------
    tuple
        A tuple containing:
            - list: List of array layouts.
            - list or None: Background layout or None if not provided.
    """
    background_layout = None
    if args_dict.get("array_layout_name_background"):
        background_layout = get_array_layouts_from_db(
            args_dict["array_layout_name_background"],
            args_dict["site"],
            args_dict["model_version"],
            args_dict["coordinate_system"],
        )["array_elements"]

    if args_dict["array_layout_name"] is not None or args_dict["plot_all_layouts"]:
        _logger.info("Plotting array from DB using layout array name(s).")
        layouts = get_array_layouts_from_db(
            args_dict["array_layout_name"],
            args_dict["site"],
            args_dict["model_version"],
            args_dict["coordinate_system"],
        )
        if isinstance(layouts, list):
            return layouts, background_layout
        return [layouts], background_layout

    if args_dict["array_layout_parameter_file"] is not None:
        _logger.info("Plotting array from parameter file(s).")
        return get_array_layouts_from_parameter_file(
            args_dict["array_layout_parameter_file"],
            args_dict["model_version"],
            args_dict["coordinate_system"],
        ), background_layout

    if args_dict["array_layout_file"] is not None:
        _logger.info("Plotting array from telescope table file(s).")
        return get_array_layouts_from_file(args_dict["array_layout_file"]), background_layout
    if args_dict["array_element_list"] is not None:
        _logger.info("Plotting array from list of array elements.")
        return get_array_layouts_using_telescope_lists_from_db(
            [args_dict["array_element_list"]],
            args_dict["site"],
            args_dict["model_version"],
            args_dict["coordinate_system"],
        ), background_layout

    return [], background_layout


def _get_array_name(array_name):
    """
    Return telescope size and number of telescopes from regular array name.

    Finetuned to array names like "4MST", "1LST", etc.

    Parameters
    ----------
    array_name : str
        Name of the regular array (e.g. "4MST").

    Returns
    -------
    tel_size : str
        Telescope size (e.g. "MST").
    n_tel : int
        Number of telescopes (e.g. 4).
    """
    if len(array_name) < 2 or not array_name[0].isdigit():
        raise ValueError(f"Invalid array_name: '{array_name}'")

    return array_name[1:], int(array_name[0])


def create_regular_array(array_name, site, telescope_distance):
    """
    Create a regular array layout table.

    Parameters
    ----------
    array_name : str
        Name of the regular array (e.g. "4MST").
    site : str
        Site identifier.
    telescope_distance : dict
        Dictionary with telescope distances per telescope type.

    Returns
    -------
    astropy.table.Table
        Table with the regular array layout.
    """
    tel_name, pos_x, pos_y, pos_z = [], [], [], []
    tel_size, n_tel = _get_array_name(array_name)
    tel_size = array_name[1:4]

    # Single telescope at the center
    if n_tel == 1:
        tel_name.append(names.generate_array_element_name_from_type_site_id(tel_size, site, "01"))
        pos_x.append(0 * u.m)
        pos_y.append(0 * u.m)
        pos_z.append(0 * u.m)
    # 4 telescopes in a regular square grid
    elif n_tel == 4:
        for i in range(1, 5):
            tel_name.append(
                names.generate_array_element_name_from_type_site_id(tel_size, site, f"0{i}")
            )
            pos_x.append(telescope_distance[tel_size] * (-1) ** (i // 2))
            pos_y.append(telescope_distance[tel_size] * (-1) ** (i % 2))
            pos_z.append(0 * u.m)
    else:
        raise ValueError(f"Unsupported number of telescopes: {n_tel}.")

    table = QTable(meta={"array_name": array_name, "site": site})
    table["telescope_name"] = tel_name
    table["position_x"] = pos_x
    table["position_y"] = pos_y
    table["position_z"] = pos_z
    table.sort("telescope_name")
    _logger.info(f"Regular array layout table:\n{table}")

    return table


def write_array_elements_from_file_to_repository(
    coordinate_system, input_file, repository_path, parameter_version
):
    """
    Read array elements from file and write their positions to model repository.

    Writes one model parameter file per array elements.

    Parameters
    ----------
    coordinate_system : str
        Coordinate system of array element positions (utm or ground).
    input_file : str or Path
        Path to input file with array element positions.
    repository_path : str or Path
        Path to model repository.
    parameter_version : str
        Parameter version to use when writing to repository.
    """
    repository_path = Path(repository_path)

    array_elements = Table.read(input_file)

    if coordinate_system == "ground":
        parameter_name = "array_element_position_ground"
        x = array_elements["position_x"].quantity.to(u.m).value
        y = array_elements["position_y"].quantity.to(u.m).value
        alt = array_elements["position_z"].quantity.to(u.m).value
    elif coordinate_system == "utm":
        x = array_elements["utm_east"].quantity.to(u.m).value
        y = array_elements["utm_north"].quantity.to(u.m).value
        alt = array_elements["altitude"].quantity.to(u.m).value
        parameter_name = "array_element_position_utm"
    else:
        raise ValueError(
            f"Unsupported coordinate system: {coordinate_system}. Allowed are 'utm' and 'ground'."
        )

    for i, row in enumerate(array_elements):
        instrument = (
            row["telescope_name"]
            if "telescope_name" in array_elements.colnames
            else f"{row['asset_code']}-{row['sequence_number']}"
        )
        output_path = repository_path / f"{instrument}"
        output_path.mkdir(parents=True, exist_ok=True)
        _logger.info(f"Writing array element positions ({coordinate_system}) to {output_path}")

        ModelDataWriter.dump_model_parameter(
            parameter_name=parameter_name,
            instrument=instrument,
            value=f"{x[i]} {y[i]} {alt[i]}",
            unit="m",
            parameter_version=parameter_version,
            output_path=repository_path / instrument,
            output_file=f"{parameter_name}.json",
        )
