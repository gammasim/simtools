"""Retrieve, merge, and write layout dictionaries."""

import ast
import logging
from pathlib import Path

import astropy.units as u
from astropy.table import QTable, Table

import simtools.utils.general as gen
import simtools.version as simtools_version
from simtools.context import RunContext
from simtools.data_model.model_data_writer import ModelDataWriter
from simtools.io import ascii_handler
from simtools.layout.array_layout_sources import (
    get_array_layouts_from_file,
    get_array_layouts_from_model_repository,
    get_array_layouts_from_parameter_file,
    get_array_layouts_using_telescope_lists_from_model_repository,
)
from simtools.utils import names

_logger = logging.getLogger(__name__)
SUBARRAY_IDS_FILE_NAME = "subarray-ids.json"


def retrieve_ctao_array_layouts(site, repository_url, branch_name="main"):
    """
    Retrieve array layouts from the CTAO common identifiers repository.

    Parameters
    ----------
    site : str
        Site identifier.
    repository_url : str
        URL or path to CTAO common identifiers.
    branch_name : str
        Repository branch to use.

    Returns
    -------
    tuple
        Array layouts for the site (list of dicts) and ``subarray-ids.json`` metadata (list).
    """
    _logger.info(f"Retrieving array layouts from {repository_url} on branch {branch_name}.")

    is_url = gen.is_url(repository_url)
    if is_url:
        array_element_ids = ascii_handler.collect_data_from_git(
            file_name="array-element-ids.json",
            git_repository=repository_url,
            git_branch=branch_name,
        )
        sub_arrays = ascii_handler.collect_data_from_git(
            file_name=SUBARRAY_IDS_FILE_NAME,
            git_repository=repository_url,
            git_branch=branch_name,
        )
    else:
        array_element_ids = ascii_handler.collect_data_from_file(
            Path(repository_url) / "array-element-ids.json"
        )
        sub_arrays = ascii_handler.collect_data_from_file(
            Path(repository_url) / SUBARRAY_IDS_FILE_NAME
        )

    layouts = _get_ctao_layouts_per_site(site, sub_arrays, array_element_ids)
    associated_data = _build_associated_data_from_metadata(
        sub_arrays.get("metadata", {}),
        url=repository_url if is_url else None,
        branch=branch_name,
    )
    return layouts, associated_data


def _build_associated_data_from_metadata(metadata, url=None, branch=None):
    """Return associated_data entries derived from subarray-ids.json metadata."""
    if product_id := metadata.get("CTA PRODUCT ID"):
        return [
            {
                "id": product_id,
                "description": "CTAO subarray identifiers",
                "filename": SUBARRAY_IDS_FILE_NAME,
                "url": url,
                "branch": branch,
            }
        ]
    return []


def _get_ctao_layouts_per_site(site, sub_arrays, array_element_ids):
    """Get array layouts for CTAO sites (sub-arrays whose elements belong to ``site``)."""
    layouts_per_site = []
    for array in sub_arrays.get("subarrays", []):
        elements = []
        for ids in array.get("array_element_ids", []):
            element_name = _get_ctao_array_element_name(ids, array_element_ids)
            if names.get_site_from_array_element_name(element_name) != site:
                break
            elements.append(element_name)
        if len(elements) > 0:
            layouts_per_site.append({"name": array.get("name"), "elements": elements})

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
        normalized_name = _normalize_array_layout_name(layout_2["name"])
        layout_found = False
        for layout_1 in layouts_1.get("value", {}):
            if sorted(layout_1["elements"]) == sorted(layout_2["elements"]):
                _logger.info(
                    f"Equal telescope list: simtools '{layout_1['name']}' "
                    f"and CTAO '{layout_2['name']}'"
                )
                layout_1["name"] = normalized_name
                layout_found = True
        if not layout_found:
            merged_layout["value"].append(
                {"name": normalized_name, "elements": layout_2["elements"]}
            )
            _logger.info(f"Adding {normalized_name} with {layout_2['elements']}")

    merged_layout["value"] = _deduplicate_and_normalize_layouts(merged_layout.get("value", []))
    return merged_layout


def _normalize_array_layout_name(name):
    """Normalize layout name by replacing blanks with dashes."""
    return str(name).replace(" ", "-")


def _deduplicate_and_normalize_layouts(layouts):
    """Deduplicate layouts by telescope set and normalize all layout names."""
    deduplicated_layouts = []
    seen_element_sets = set()
    for layout in layouts:
        element_key = tuple(sorted(layout.get("elements", [])))
        if element_key in seen_element_sets:
            _logger.info(
                f"Skipping duplicate layout {layout.get('name')} with elements {element_key}"
            )
            continue
        deduplicated_layouts.append(
            {
                "name": _normalize_array_layout_name(layout.get("name")),
                "elements": layout.get("elements", []),
            }
        )
        seen_element_sets.add(element_key)
    return deduplicated_layouts


def write_array_layouts(array_layouts, args_dict, associated_data=None, *, context=None):
    """
    Write array layouts as model parameter.

    Parameters
    ----------
    array_layouts : dict
        Array layouts to be written.
    args_dict : dict
        Command line arguments.
    associated_data : list, optional
        Associated data entries to include in the metadata.
    context : RunContext, optional
        Run context (defaults to the environment).
    """
    site = args_dict.get("site") or array_layouts.get("site")
    _logger.info(f"Writing updated array layouts to the model repository for site {site}.")

    context = context or RunContext.from_environment()
    output_file = context.paths.get_output_file(
        f"array_layouts-{args_dict['updated_parameter_version']}.json",
        sub_dir="array_layouts",
    )

    metadata_input_dict = dict(args_dict)
    for entry in associated_data or []:
        metadata_input_dict.setdefault("associated_data", []).append(entry)

    ModelDataWriter.write_model_parameter(
        parameter_name="array_layouts",
        value=array_layouts["value"],
        instrument=f"OBS-{site}",
        parameter_version=args_dict.get("updated_parameter_version"),
        output_file=output_file,
        metadata_input_dict=metadata_input_dict,
        context=context,
    )


def validate_array_layouts_with_model_repository(production_table, array_layouts):
    """
    Validate that all telescopes in the array layouts exist in the production table.

    Parameters
    ----------
    production_table : dict
        Production table from the model repository.
    array_layouts : dict
        Array layouts to be validated.

    Returns
    -------
    dict
        Validated array layouts.

    Raises
    ------
    ValueError
        If array elements are not in the production table.
    """
    model_elements = set(production_table.get("parameters", {}).keys())
    invalid_array_elements = [
        e
        for layout in array_layouts.get("value", [])
        for e in layout.get("elements", [])
        if e not in model_elements
    ]
    if invalid_array_elements:
        raise ValueError(f"Invalid array elements found: {invalid_array_elements}. ")
    return array_layouts


def get_array_layout_elements(array_layouts, layout_name):
    """
    Return the elements of a named layout in an array-layout parameter.

    Parameters
    ----------
    array_layouts : dict
        Canonical ``array_layouts`` model parameter.
    layout_name : str
        Name of the reference layout.

    Returns
    -------
    list
        Elements in the named layout, in stored order.

    Raises
    ------
    ValueError
        If the layout does not exist or has no elements.
    """
    for layout in array_layouts.get("value", []):
        if layout.get("name") == layout_name:
            elements = layout.get("elements", [])
            if not elements:
                raise ValueError(f"Reference array layout '{layout_name}' has no elements.")
            return list(elements)
    raise ValueError(f"Reference array layout '{layout_name}' not found.")


def validate_array_layout_subset_of_reference(
    layout_name, elements, reference_elements, reference_layout_name
):
    """
    Validate and return one array layout constrained by a reference layout.

    Parameters
    ----------
    layout_name : str
        Name of the new layout.
    elements : list
        Telescope names selected for the new layout.
    reference_elements : list
        Telescope names allowed by the reference layout.
    reference_layout_name : str
        Name used in validation error messages.

    Returns
    -------
    dict
        Canonical one-layout dictionary.

    Raises
    ------
    ValueError
        If the name or element list is invalid, has duplicates, or is not a subset
        of the reference layout.
    """
    if not isinstance(layout_name, str) or not layout_name.strip():
        raise ValueError("Array layout name must be a non-empty string.")
    if not isinstance(elements, list) or not elements:
        raise ValueError(f"Array layout '{layout_name}' requires a non-empty telescope list.")
    if len(elements) != len(set(elements)):
        raise ValueError(f"Array layout '{layout_name}' contains duplicate telescopes.")

    reference_set = set(reference_elements)
    invalid_elements = [element for element in elements if element not in reference_set]
    if invalid_elements:
        raise ValueError(
            f"Array layout '{layout_name}' contains elements outside reference layout "
            f"'{reference_layout_name}': {invalid_elements}."
        )
    return {"name": layout_name, "elements": list(elements)}


def prepare_array_layouts_for_submission(model_reader, args_dict):
    """
    Read or construct array layouts for the submit application.

    The legacy file input is returned unchanged. Direct input reads the base
    parameter, validates the requested layout against its reference layout, and
    adds the new layout to the base value.

    Parameters
    ----------
    model_reader : SimulationModelReader
        Configured source-neutral simulation-model reader.
    args_dict : dict
        Parsed application arguments.

    Returns
    -------
    tuple
        Array-layout parameter and the single model version used for validation.

    Raises
    ------
    ValueError
        If the input mode or required direct-layout arguments are invalid.
    """
    file_input = args_dict.get("array_layouts")
    layout_name = args_dict.get("array_layout_name")
    element_list = args_dict.get("array_element_list")
    if file_input:
        if layout_name or element_list:
            raise ValueError(
                "Use either array_layouts or array_layout_name/array_element_list, not both."
            )
        return ascii_handler.collect_data_from_file(file_input), _get_single_model_version(
            args_dict.get("model_version")
        )

    missing = [
        name
        for name in (
            "array_layout_name",
            "array_element_list",
            "site",
            "model_version",
            "parameter_version",
            "updated_parameter_version",
        )
        if not args_dict.get(name)
    ]
    if missing:
        raise ValueError("Direct layout input requires: " + ", ".join(missing))

    model_version = _get_single_model_version(args_dict["model_version"])
    base_layouts = model_reader.get_model_parameter(
        parameter="array_layouts",
        site=args_dict["site"],
        array_element_name=None,
        parameter_version=args_dict["parameter_version"],
    )["array_layouts"]
    parameter_site = base_layouts.get("site")
    if parameter_site and parameter_site != args_dict["site"]:
        raise ValueError(
            f"Array-layout parameter site '{parameter_site}' does not match requested site "
            f"'{args_dict['site']}'."
        )

    reference_name = args_dict.get("reference_array_layout", "hyper_array")
    new_layout = validate_array_layout_subset_of_reference(
        layout_name=layout_name,
        elements=element_list,
        reference_elements=get_array_layout_elements(base_layouts, reference_name),
        reference_layout_name=reference_name,
    )
    normalized_name = _normalize_array_layout_name(new_layout["name"])
    existing_names = {
        _normalize_array_layout_name(layout.get("name")) for layout in base_layouts.get("value", [])
    }
    if normalized_name in existing_names:
        raise ValueError(f"Array layout '{new_layout['name']}' already exists.")

    base_layouts.setdefault("value", []).append(
        {"name": normalized_name, "elements": new_layout["elements"]}
    )
    return base_layouts, model_version


def _get_single_model_version(model_version):
    """Return the single model version accepted by array-layout submission."""
    if isinstance(model_version, list):
        if len(model_version) != 1:
            raise ValueError("submit-array-layouts accepts exactly one model version.")
        return model_version[0]
    return model_version


def read_layouts(args_dict, model_reader=None, context=None):
    """
    Read array layouts from the model repository or parameter file.

    Parameters
    ----------
    args_dict : dict
        Dictionary with command line arguments.
    model_reader : SimulationModelReader, optional
        Model reader.
    context : RunContext, optional
        Run context.

    Returns
    -------
    tuple
        List of array layouts and the background layout (or None).

    Raises
    ------
    ValueError
        If array_layout_name_from_parameter_file is given without array_layout_parameter_file.
    """
    if args_dict.get("array_layout_name_from_parameter_file") and not args_dict.get(
        "array_layout_parameter_file"
    ):
        raise ValueError(
            "array_layout_name_from_parameter_file requires array_layout_parameter_file."
        )

    background_layout = None
    ignore_software_version = args_dict.get("ignore_software_version", False)
    if args_dict.get("array_layout_name_background"):
        background_layout = get_array_layouts_from_model_repository(
            args_dict["array_layout_name_background"],
            args_dict["site"],
            args_dict["model_version"],
            args_dict["coordinate_system"],
            ignore_software_version,
            model_reader=model_reader,
            context=context,
        )["array_elements"]

    if args_dict["array_layout_parameter_file"] is not None:
        _logger.info("Plotting array from parameter file(s).")
        return get_array_layouts_from_parameter_file(
            args_dict["array_layout_parameter_file"],
            args_dict["model_version"],
            args_dict["coordinate_system"],
            args_dict.get("array_layout_name_from_parameter_file"),
            context=context,
        ), background_layout

    if args_dict["array_layout_name"] is not None or args_dict["plot_all_layouts"]:
        _logger.info("Plotting array from the model repository using layout array name(s).")
        layouts = get_array_layouts_from_model_repository(
            args_dict["array_layout_name"],
            args_dict["site"],
            args_dict["model_version"],
            args_dict["coordinate_system"],
            ignore_software_version,
            model_reader=model_reader,
            context=context,
        )
        if isinstance(layouts, list):
            return layouts, background_layout
        return [layouts], background_layout

    if args_dict["array_layout_file"] is not None:
        _logger.info("Plotting array from telescope table file(s).")
        return (
            get_array_layouts_from_file(args_dict["array_layout_file"], context=context),
            background_layout,
        )
    if args_dict["array_element_list"] is not None:
        _logger.info("Plotting array from list of array elements.")
        return get_array_layouts_using_telescope_lists_from_model_repository(
            [args_dict["array_element_list"]],
            args_dict["site"],
            args_dict["model_version"],
            args_dict["coordinate_system"],
            ignore_software_version,
            model_reader=model_reader,
            context=context,
        ), background_layout

    return [], background_layout


def _create_star_array(tel_name, pos_x, pos_y, pos_z, n_telescopes, tel_type, site, distance):
    """Create star-shaped array positions along the x and y axes."""
    axis_sequence = ["x", "y", "-x", "-y"]
    step = 1
    for i in range(n_telescopes):
        tel_name.append(
            names.generate_array_element_name_from_type_site_id(tel_type, site, f"{i + 1:02d}")
        )
        dist = distance * step
        x, y = {
            "x": (dist, 0 * u.m),
            "-x": (-dist, 0 * u.m),
            "y": (0 * u.m, dist),
            "-y": (0 * u.m, -dist),
        }[axis_sequence[i % 4]]
        pos_x.append(x)
        pos_y.append(y)
        pos_z.append(0 * u.m)
        if (i + 1) % 4 == 0:
            step += 1


def _create_square_array(tel_name, pos_x, pos_y, pos_z, n_tel, tel_type, site, distance):
    """Create square array positions (1 or 4 telescopes)."""
    if n_tel == 1:
        tel_name.append(names.generate_array_element_name_from_type_site_id(tel_type, site, "01"))
        pos_x.append(0 * u.m)
        pos_y.append(0 * u.m)
        pos_z.append(0 * u.m)
    elif n_tel == 4:
        for i in range(1, 5):
            tel_name.append(
                names.generate_array_element_name_from_type_site_id(tel_type, site, f"0{i}")
            )
            pos_x.append(distance * (-1) ** (i // 2))
            pos_y.append(distance * (-1) ** (i % 2))
            pos_z.append(0 * u.m)
    else:
        raise ValueError(f"Unsupported number of telescopes for square array: {n_tel}.")


def create_regular_array(
    array_name, site, n_telescopes, telescope_type, telescope_distance, shape="square"
):
    """
    Create a regular array layout table.

    Parameters
    ----------
    array_name : str
        Name of the regular array (e.g. "4MST").
    site : str
        Site identifier.
    n_telescopes : int
        Number of telescopes in the array.
    telescope_type : str
        Type of telescope (e.g. "MST").
    telescope_distance : Quantity
        Distance between telescopes in the array.
    shape : str
        Shape of the array: "square" or "star" (default: "square").

    Returns
    -------
    astropy.table.Table
        Table with the regular array layout.

    Raises
    ------
    ValueError
        If the shape is not supported.
    """
    builders = {"square": _create_square_array, "star": _create_star_array}
    if shape not in builders:
        raise ValueError(f"Unsupported array shape: {shape}. Allowed: 'square' or 'star'.")

    tel_name, pos_x, pos_y, pos_z = [], [], [], []
    builders[shape](
        tel_name, pos_x, pos_y, pos_z, n_telescopes, telescope_type, site, telescope_distance
    )

    table = QTable(meta={"array_name": array_name, "site": site})
    table["telescope_name"] = tel_name
    table["position_x"] = pos_x
    table["position_y"] = pos_y
    table["position_z"] = pos_z
    table.sort("telescope_name")
    _logger.info(f"Regular array layout table:\n{table}")
    return table


def write_array_elements_from_file_to_repository(
    coordinate_system, input_file, repository_path, parameter_version, context=None
):
    """
    Read array elements from file and write their positions to the model repository.

    Writes one model parameter file per array element.

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
    context : RunContext, optional
        Run context.

    Notes
    -----
    Writes files directly and does not query the model repository for an existing version.
    """
    repository_path = Path(repository_path)
    array_elements = Table.read(input_file)

    if coordinate_system == "ground":
        parameter_name = "array_element_position_ground"
        columns = ("position_x", "position_y", "position_z")
    elif coordinate_system == "utm":
        parameter_name = "array_element_position_utm"
        columns = ("utm_east", "utm_north", "altitude")
    else:
        raise ValueError(
            f"Unsupported coordinate system: {coordinate_system}. Allowed are 'utm' and 'ground'."
        )
    x, y, alt = (array_elements[c].quantity.to(u.m).value for c in columns)

    for i, row in enumerate(array_elements):
        instrument = (
            row["telescope_name"]
            if "telescope_name" in array_elements.colnames
            else f"{row['asset_code']}-{row['sequence_number']}"
        )
        output_path = repository_path / f"{instrument}"
        output_path.mkdir(parents=True, exist_ok=True)
        _logger.info(f"Writing array element positions ({coordinate_system}) to {output_path}")

        ModelDataWriter.write_model_parameter(
            parameter_name=parameter_name,
            instrument=instrument,
            value=f"{x[i]} {y[i]} {alt[i]}",
            unit="m",
            parameter_version=parameter_version,
            output_path=output_path,
            output_file=f"{parameter_name}.json",
            check_for_existing_parameter=False,
            context=context,
        )


def write_array_elements_info_yaml(
    array_table, site, model_version, output_file, parameter_version="2.0.0"
):
    """
    Write YAML file for array layout in a format compatible with overwrite.

    Parameters
    ----------
    array_table : astropy.table.Table
        Table with telescope positions.
    site : str
        Site identifier (e.g., 'North' or 'South').
    model_version : str
        Model version.
    output_file : Path
        Path to the output YAML file.
    parameter_version : str
        Parameter version to use in the YAML file.
    """
    telescope_names = [str(name) for name in array_table["telescope_name"]]
    array_layout = {
        "version": parameter_version,
        "value": [{"name": array_table.meta["array_name"], "elements": telescope_names}],
        "unit": None,
    }
    data = {
        "model_version": model_version,
        "model_update": "patch_update",
        "model_version_history": [model_version],
        "description": f"Regular array layout: {array_table.meta['array_name']}",
        "changes": {f"OBS-{site}": {"array_layouts": array_layout}},
    }

    for tel_name, pos_x, pos_y, pos_z in zip(
        telescope_names,
        array_table["position_x"],
        array_table["position_y"],
        array_table["position_z"],
    ):
        data["changes"][tel_name] = {
            "array_element_position_ground": {
                "version": parameter_version,
                "value": [
                    float(pos_x.to(u.m).value),
                    float(pos_y.to(u.m).value),
                    float(pos_z.to(u.m).value),
                ],
                "unit": "m",
            }
        }

    ascii_handler.write_data_to_file(data, output_file)


def resolve_array_layout_name(array_layout_name, model_version):
    """
    Resolve array layout configuration for a specific model version.

    Parameters
    ----------
    array_layout_name : str or dict or list of str or list of dict
        Array layout name(s) or configuration(s) to resolve.
    model_version : str
        Model version to resolve for.

    Returns
    -------
    str or dict or list
        The resolved array layout name(s) or configuration.
    """
    if isinstance(array_layout_name, list) and len(array_layout_name) == 1:
        array_layout_name = array_layout_name[0]

    if isinstance(array_layout_name, str) and array_layout_name.strip().startswith("{"):
        try:
            parsed_layout = ast.literal_eval(array_layout_name)
            if isinstance(parsed_layout, dict):
                array_layout_name = parsed_layout
        except SyntaxError, ValueError:
            return array_layout_name

    if not isinstance(array_layout_name, dict) or list(array_layout_name) != ["by_version"]:
        return array_layout_name

    resolved = simtools_version.resolve_by_version(
        {"array_layout_name": array_layout_name}, model_version
    )
    return resolved["array_layout_name"]
