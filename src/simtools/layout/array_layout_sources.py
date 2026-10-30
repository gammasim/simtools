"""Read array layouts from the model repository, parameter files, and astropy table files."""

from pathlib import Path

from simtools.data_model import data_reader
from simtools.io import ascii_handler
from simtools.model.array_model import ArrayModel
from simtools.model.site_model import SiteModel
from simtools.utils import names, value_conversion


def get_array_layouts_from_parameter_file(
    file_path, model_version, coordinate_system="ground", array_layout_name=None, context=None
):
    """
    Retrieve array layouts from a parameter file.

    Parameters
    ----------
    file_path : str or Path
        Path to the array layout parameter file.
    model_version : str
        Model version to retrieve.
    coordinate_system : str
        Coordinate system for the array elements (default: "ground").
    array_layout_name : list, optional
        Layouts to retrieve (None: retrieve all).
    context : RunContext, optional
        Run context.

    Returns
    -------
    list
        List of dictionaries containing array layout names and their elements.

    Raises
    ------
    ValueError
        If the file has no 'value' key.
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
            context=context,
        )
        for layout in value
        if array_layout_name is None or layout.get("name") in array_layout_name
    ]


def get_array_layouts_from_model_repository(
    layout_name,
    site,
    model_version,
    coordinate_system="ground",
    ignore_software_version=False,
    model_reader=None,
    context=None,
):
    """
    Retrieve array layouts from the model repository.

    Parameters
    ----------
    layout_name : str or list
        Layout name(s) to retrieve (None: retrieve all layouts).
    site : str
        Site identifier.
    model_version : str
        Model version to retrieve.
    coordinate_system : str
        Coordinate system for the array elements (default: "ground").
    ignore_software_version : bool
        If True, skip the software version compatibility check.
    model_reader : SimulationModelReader, optional
        Model reader.
    context : RunContext, optional
        Run context.

    Returns
    -------
    list or dict
        Array layouts (a single dict if exactly one layout is found).
    """
    if layout_name:
        layout_names = value_conversion.ensure_list(layout_name)
    else:
        site_model = SiteModel(
            site=site,
            model_version=model_version,
            ignore_software_version=ignore_software_version,
            model_reader=model_reader,
            context=context,
        )
        layout_names = site_model.get_list_of_array_layouts()

    layouts = [
        _get_array_layout_dict(
            model_version,
            site,
            None,
            _layout_name,
            coordinate_system,
            ignore_software_version,
            model_reader=model_reader,
            context=context,
        )
        for _layout_name in layout_names
    ]
    if len(layouts) == 1:
        return layouts[0]
    return layouts


def get_array_layouts_using_telescope_lists_from_model_repository(
    telescope_lists,
    site,
    model_version,
    coordinate_system="ground",
    ignore_software_version=False,
    model_reader=None,
    context=None,
):
    """
    Retrieve array layouts from the model repository using telescope lists.

    Parameters
    ----------
    telescope_lists : list
        Telescope lists to retrieve array layouts for.
    site : str or None
        Site identifier. If None, derived from the telescope names.
    model_version : str
        Model version to retrieve.
    coordinate_system : str
        Coordinate system for the array elements (default: "ground").
    ignore_software_version : bool
        If True, skip the software version compatibility check.
    model_reader : SimulationModelReader, optional
        Model reader.
    context : RunContext, optional
        Run context.

    Returns
    -------
    list
        List of dictionaries containing array layout names and their elements.

    Raises
    ------
    ValueError
        If a telescope list mixes sites and no site is given.
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
                model_version,
                _site,
                telescope_list,
                None,
                coordinate_system,
                ignore_software_version,
                model_reader,
                context=context,
            )
        )
    return layouts


def get_array_layouts_from_file(file_path, context=None):
    """
    Retrieve array layout(s) from astropy table file(s).

    Parameters
    ----------
    file_path : str or Path or list of str or list of Path
        Path(s) to array layout file(s).
    context : RunContext, optional
        Run context.

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
            "array_elements": data_reader.read_table_from_file(file_name=_file, context=context),
        }
        for _file in file_path
    ]


def _get_array_layout_dict(
    model_version,
    site,
    telescope_list,
    layout_name,
    coordinate_system,
    ignore_software_version=False,
    model_reader=None,
    *,
    context,
):
    """Return array layout dictionary for a given telescope list."""
    array_model = ArrayModel(
        model_version=model_version,
        site=site,
        array_elements=telescope_list,
        layout_name=layout_name,
        ignore_software_version=ignore_software_version,
        model_reader=model_reader,
        context=context,
    )
    return {
        "name": layout_name if layout_name else "list",
        "site": site,
        "array_elements": array_model.export_array_elements_as_table(
            coordinate_system=coordinate_system
        ),
    }


def get_array_elements_from_model_repository(
    layouts, site, model_version, model_reader=None, context=None
):
    """
    Get the array elements from the model repository for a list of layout names.

    Structure of the returned dictionary::

        {"layout_name_1": [telescope_id_1, ...], "layout_name_2": [telescope_id_3, ...]}

    Parameters
    ----------
    layouts : list[str]
        Layout names to read. If ["all"], read all available layouts.
    site : str
        Site name for the array layouts.
    model_version : str
        Model version for the array layouts.
    model_reader : SimulationModelReader, optional
        Model reader.
    context : RunContext, optional
        Run context.

    Returns
    -------
    dict
        Dictionary mapping layout names to telescope IDs.
    """
    site_model = SiteModel(
        site=site,
        model_version=model_version,
        model_reader=model_reader,
        context=context,
    )
    layout_names = site_model.get_list_of_array_layouts() if layouts == ["all"] else layouts
    return {
        layout_name: site_model.get_array_elements_for_layout(layout_name)
        for layout_name in layout_names
    }
