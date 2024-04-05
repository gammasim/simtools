import logging
import re

_logger = logging.getLogger(__name__)

__all__ = [
    "get_site_from_telescope_name",
    "get_telescope_type_from_telescope_name",
    "generate_file_name",
    "layout_telescope_list_file_name",
    "sanitize_name",
    "simtel_single_mirror_list_file_name",
    "simtel_config_file_name",
    "validate_array_layout_name",
    "validate_model_version_name",
    "validate_site_name",
    "validate_telescope_id_name",
    "validate_telescope_name",
]

# Telescopes and other array elements
array_element_names = {
    # CTAO telescopes
    "LSTN": {"site": "North", "observatory": "CTAO", "class": "telescope"},
    "MSTN": {"site": "North", "observatory": "CTAO", "class": "telescope"},
    "LSTS": {"site": "South", "observatory": "CTAO", "class": "telescope"},
    "MSTS": {"site": "South", "observatory": "CTAO", "class": "telescope"},
    "SSTS": {"site": "South", "observatory": "CTAO", "class": "telescope"},
    "SCTS": {"site": "South", "observatory": "CTAO", "class": "telescope"},
    # calibration devices
    "ILLN": {"site": "North", "observatory": "CTAO", "class": "calibration"},
    "RLDN": {"site": "North", "observatory": "CTAO", "class": "calibration"},
    "STPN": {"site": "North", "observatory": "CTAO", "class": "calibration"},
    "MSPN": {"site": "North", "observatory": "CTAO", "class": "calibration"},
    "CEIN": {"site": "North", "observatory": "CTAO", "class": "calibration"},
    "WSTN": {"site": "North", "observatory": "CTAO", "class": "calibration"},
    "ASCN": {"site": "North", "observatory": "CTAO", "class": "calibration"},
    "DUSN": {"site": "North", "observatory": "CTAO", "class": "calibration"},
    "LISN": {"site": "North", "observatory": "CTAO", "class": "calibration"},
    "ILLS": {"site": "South", "observatory": "CTAO", "class": "calibration"},
    "RLDS": {"site": "South", "observatory": "CTAO", "class": "calibration"},
    "STPS": {"site": "South", "observatory": "CTAO", "class": "calibration"},
    "MSPS": {"site": "South", "observatory": "CTAO", "class": "calibration"},
    "CEIS": {"site": "South", "observatory": "CTAO", "class": "calibration"},
    "WSTS": {"site": "South", "observatory": "CTAO", "class": "calibration"},
    "ASCS": {"site": "South", "observatory": "CTAO", "class": "calibration"},
    "DUSS": {"site": "South", "observatory": "CTAO", "class": "calibration"},
    "LISS": {"site": "South", "observatory": "CTAO", "class": "calibration"},
    # other telescopes
    "MAGIC": {"site": "North", "observatory": "MAGIC", "class": "telescope"},
    "VERITAS": {"site": "North", "observatory": "VERITAS", "class": "telescope"},
    "HESS": {"site": "South", "observatory": "HESS", "class": "telescope"},
}

site_names = {
    "South": ["paranal", "south", "cta-south", "ctao-south", "s"],
    "North": ["lapalma", "north", "cta-north", "ctao-north", "n"],
}

model_version_names = {
    "2015-07-21": [""],
    "2015-10-20-p1": [""],
    "prod4-v0.0": [""],
    "prod4-v0.1": [""],
    "2018-02-16": [""],
    "prod3_compatible": ["p3", "prod3", "prod3b"],
    "prod4": ["p4"],
    "post_prod3_updates": [""],
    "2016-12-20": [""],
    "2018-11-07": [""],
    "2019-02-22": [""],
    "2019-05-13": [""],
    "2019-11-20": [""],
    "2019-12-30": [""],
    "2020-02-26": [""],
    "2020-06-28": ["prod5"],
    "2024-02-01": ["prod6"],
    "prod4-prototype": [""],
    "default": [],
    "Released": [],
    "Latest": [],
}

array_layout_names = {
    "4LST": ["4-lst", "4lst"],
    "1LST": ["1-lst", "1lst"],
    "4MST": ["4-mst", "4mst"],
    "1MST": ["1-mst", "mst"],
    "4SST": ["4-sst", "4sst"],
    "1SST": ["1-sst", "sst"],
    "Prod5": ["prod5", "p5"],
    "TestLayout": ["test-layout"],
}

# TODO - this is temporary
# List of site parameters which are not part of the simtel configuration
# or which have different naming in the database and simtel configuration.
# simtel: True if this is a simtel parameter (allows to give alternative "name")
site_parameters = {
    "reference_point_altitude": {"db_name": "altitude", "simtel": False},
    "reference_point_longitude": {"db_name": "ref_long", "simtel": False},
    "reference_point_latitude": {"db_name": "ref_lat", "simtel": False},
    "reference_point_utm_north": {"db_name": "reference_point_utm_north", "simtel": False},
    "reference_point_utm_east": {"db_name": "reference_point_utm_east", "simtel": False},
    # Note naming inconsistency between old and new model
    # altitude was the corsika observation level in the old model
    "corsika_observation_level": {"db_name": "altitude", "simtel": True},
    "epsg_code": {"db_name": "epsg_code", "simtel": False},
    "geomag_horizontal": {"db_name": "geomag_horizontal", "simtel": False},
    "geomag_vertical": {"db_name": "geomag_vertical", "simtel": False},
    "geomag_rotation": {"db_name": "geomag_rotation", "simtel": False},
    "array_coordinates": {"db_name": "array_coordinates", "simtel": False},
    "atmospheric_profile": {"db_name": "atmospheric_profile", "simtel": False},
    # TODO Duplication of old names; requires renaming in DB
    "magnetic_field": {"db_name": "magnetic_field", "simtel": False},
    "EPSG": {"db_name": "EPSG", "simtel": False},
    "ref_long": {"db_name": "ref_long", "simtel": False},
    "ref_lat": {"db_name": "ref_lat", "simtel": False},
    "nsb_reference_value": {"db_name": "nsb_reference_value", "simtel": False},
}

# TODO - this is temporary
# List of telescope parameters which are not part of the simtel configuration
# or which has a different name in the simtel configuration.
telescope_parameters = {
    "telescope_axis_height": {"db_name": "telescope_axis_height", "simtel": False},
    "telescope_sphere_radius": {"db_name": "telescope_sphere_radius", "simtel": False},
    "pixel_shape": {"db_name": "pixel_shape", "simtel": False},
    "pixel_diameter": {"db_name": "pixel_diameter", "simtel": False},
    "lightguide_efficiency_vs_incident_angle": {
        "db_name": "lightguide_efficiency_vs_incident_angle",
        "simtel": False,
    },
    "lightguide_efficiency_vs_wavelength": {
        "db_name": "lightguide_efficiency_vs_wavelength",
        "simtel": False,
    },
    "mirror_panel_shape": {"db_name": "mirror_panel_shape", "simtel": False},
    "mirror_panel_diameter": {"db_name": "mirror_panel_diameter", "simtel": False},
    "asum_shaping": {"db_name": "asum_shaping_file", "simtel": True},
    "dsum_shaping": {"db_name": "dsum_shaping_file", "simtel": True},
    "nsb_pixel_rate": {"db_name": "nsb_pixel_rate", "simtel": False},
    "nsb_reference_value": {"db_name": "nsb_reference_value", "simtel": False},
    "primary_mirror_diameter": {"db_name": "primary_diameter", "simtel": True},
    "primary_mirror_degraded_map": {"db_name": "primary_degraded_map", "simtel": True},
    "primary_mirror_hole_diameter": {"db_name": "primary_hole_diameter", "simtel": True},
    "primary_mirror_ref_radius": {"db_name": "primary_ref_radius", "simtel": True},
    "primary_mirror_segmentation": {"db_name": "primary_segmentation", "simtel": True},
    "secondary_mirror_baffle": {"db_name": "secondary_baffle", "simtel": True},
    "secondary_mirror_degraded_map": {"db_name": "secondary_degraded_map", "simtel": True},
    "secondary_mirror_degraded_reflection": {
        "db_name": "mirror2_degraded_reflection",
        "simtel": True,
    },
    "secondary_mirror_diameter": {"db_name": "secondary_diameter", "simtel": True},
    "secondary_mirror_hole_diameter": {"db_name": "secondary_hole_diameter", "simtel": True},
    "secondary_mirror_ref_radius": {"db_name": "secondary_ref_radius", "simtel": True},
    "secondary_mirror_reflectivity": {"db_name": "mirror_secondary_reflectivity", "simtel": True},
    "secondary_mirror_segmentation": {"db_name": "secondary_segmentation", "simtel": True},
    "secondary_mirror_shadow_diameter": {"db_name": "secondary_shadow_diameter", "simtel": True},
    "secondary_mirror_shadow_offset": {"db_name": "secondary_shadow_offset", "simtel": True},
    "camera_filter_incidence_angle": {"db_name": "camera_filter_incidence_angle", "simtel": False},
    "camera_window_incidence_angle": {"db_name": "camera_window_incidence_angle", "simtel": False},
    "primary_mirror_incidence_angle": {
        "db_name": "primary_mirror_incidence_angle",
        "simtel": False,
    },
    "secondary_mirror_incidence_angle": {
        "db_name": "secondary_mirror_incidence_angle",
        "simtel": False,
    },
}


def validate_telescope_id_name(name):
    """
    Validate telescope ID. Allowed IDs are
    - design (for design telescopes or testing)
    - telescope ID (e.g., 1, 5, 15)
    - test (for testing)

    Parameters
    ----------
    name: str or int
        Telescope ID name.

    Returns
    -------
    str
        Validated telescope ID (added leading zeros, e.g., 1 is converted to 01).

    Raises
    ------
    ValueError
        If name is not valid.
    """

    if isinstance(name, int) or name.isdigit():
        return f"{int(name):02d}"
    if name.lower() in ("design", "test"):
        return str(name).lower()

    msg = f"Invalid telescope ID name {name}"
    _logger.error(msg)
    raise ValueError(msg)


def validate_model_version_name(name):
    """
    Validate model version name.

    Parameters
    ----------
    name: str
        Model version name.

    Returns
    -------
    str
        Validated name.
    """
    return _validate_name(name, model_version_names)


def validate_site_name(name):
    """
    Validate site name.

    Parameters
    ----------
    name: str
        Site name.

    Returns
    -------
    str
        Validated name.
    """
    return _validate_name(name, site_names)


def validate_array_layout_name(name):
    """
    Validate array layout name.

    Parameters
    ----------
    name: str
        Layout array name.

    Returns
    -------
    str
        Validated name.
    """
    return _validate_name(name, array_layout_names)


def _validate_name(name, all_names):
    """
    Validate name given the all_names options. For each key in all_names, a list of options is \
    given. If name is in this list, the key name is returned.

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


def validate_telescope_name(name):
    """
    Validate telescope name (e.g., MSTN-design, MSTN-01).

    Parameters
    ----------
    name: str
        Telescope name.

    Returns
    -------
    str
        Validated name.
    """
    try:
        _tel_type, _tel_id = name.split("-")
    except ValueError as exc:
        msg = f"Invalid name {name}"
        raise ValueError(msg) from exc
    return (
        _validate_name(_tel_type, array_element_names) + "-" + validate_telescope_id_name(_tel_id)
    )


def get_telescope_name_from_type_site_id(telescope_type, site, telescope_id):
    """
    Get telescope name from type, site and ID.

    Parameters
    ----------
    telescope_type: str
        Telescope type.
    site: str
        Site name.
    telescope_id: str
        Telescope ID.

    Returns
    -------
    str
        Telescope name.
    """
    _short_site = validate_site_name(site)[0]
    _val_id = validate_telescope_id_name(telescope_id)
    return f"{telescope_type}{_short_site}-{_val_id}"


def get_telescope_type_from_telescope_name(name):
    """
    Get telescope type from name, e.g. "LSTN", "MSTN".

    Parameters
    ----------
    telescope_name: str
        Telescope name

    Returns
    -------
    str
        Telescope type.
    """
    return _validate_name(name.split("-")[0], array_element_names)


def get_list_of_telescope_types(array_element_class="telescope", site=None, observatory="CTAO"):
    """
    Get list of telescope types.

    Parameters
    ----------
    array_element_class: str
        Array element class
    site: str
        Site name (e.g., South or North).

    Returns
    -------
    list
        List of telescope types.
    """
    return [
        key
        for key, value in array_element_names.items()
        if value["class"] == array_element_class
        and (site is None or value["site"] == site)
        and (observatory is None or value["observatory"] == observatory)
    ]


def get_site_from_telescope_name(name):
    """
    Get site name from telescope name.

    Parameters
    ----------
    name: str
        Telescope name.

    Returns
    -------
    str
        Site name (South or North).
    """
    return array_element_names[get_telescope_type_from_telescope_name(name)]["site"]


def get_class_from_telescope_name(name):
    """
    Get class (e.g., telescope, calibration) of array element from name

    Parameters
    ----------
    name: str
        Telescope name.

    Returns
    -------
    str
        Class name.
    """

    return array_element_names[get_telescope_type_from_telescope_name(name)]["class"]


def get_simtel_name_from_parameter_name(
    par_name, search_telescope_parameters=True, search_site_parameters=True
):
    """
    Get the simtel parameter name from the model parameter name.
    Assumes that both names are equal if not defined otherwise in names.py
    Returns the model parameter name if no simtel name is found.

    Parameters
    ----------
    par_name: str
        Model parameter name.
    search_telescope_parameters: bool
        If True, telescope model parameters are included.
    search_site_parameters: bool
        If True, site model parameters are included.

    Returns
    -------
    str
        Simtel parameter name.
    """

    _parameter_names = {}
    if search_telescope_parameters:
        _parameter_names.update(telescope_parameters)
    if search_site_parameters:
        _parameter_names.update(site_parameters)

    try:
        return (
            _parameter_names[par_name]["db_name"] if _parameter_names[par_name]["simtel"] else None
        )
    except KeyError:
        pass
    return par_name


def get_parameter_name_from_simtel_name(simtel_name):
    """
    Get the model parameter name from the simtel parameter name.
    Assumes that both names are equal if not defined otherwise in names.py.

    Parameters
    ----------
    simtel_name: str
        Simtel parameter name.

    Returns
    -------
    str
        Model parameter name.
    """

    _parameter_names = {**telescope_parameters, **site_parameters}

    for par_name, par_info in _parameter_names.items():
        if par_info.get("db_name") == simtel_name and par_info.get("simtel"):
            return par_name
    return simtel_name


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


def layout_telescope_list_file_name(name, label):
    """
    File name for files required at the RayTracing class.

    Parameters
    ----------
    name: str
        Name of the array.
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    file_name = f"telescope_positions-{name}"
    file_name += f"_{label}" if label is not None else ""
    file_name += ".ecsv"
    return file_name


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
    Used e.g., to generate camera-efficiency and ray-tracing output files.

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
    name = f"{file_type}-{site}-{telescope_model_name}"
    name += f"-d{source_distance:.1f}km" if source_distance is not None else ""
    name += f"-za{float(zenith_angle):.1f}deg"
    name += f"-off{off_axis_angle:.3f}deg" if off_axis_angle is not None else ""
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
        if the string `name` can not be sanitized.
    """

    # Convert to lowercase
    sanitized = name.lower()

    # Replace spaces with underscores
    sanitized = sanitized.replace(" ", "_")

    # Remove characters that are not alphanumerics or underscores
    sanitized = re.sub(r"\W|^(?=\d)", "_", sanitized)
    if not sanitized.isidentifier():
        msg = f"The string {name} could not be sanitized."
        _logger.error(msg)
        raise ValueError
    return sanitized
