import logging
import re

_logger = logging.getLogger(__name__)

__all__ = [
    "camera_efficiency_log_file_name",
    "camera_efficiency_results_file_name",
    "camera_efficiency_simtel_file_name",
    "get_site_from_telescope_name",
    "get_telescope_type_from_telescope_name",
    "layout_telescope_list_file_name",
    "ray_tracing_file_name",
    "ray_tracing_plot_file_name",
    "ray_tracing_results_file_name",
    "sanitize_name",
    "simtel_array_config_file_name",
    "simtel_single_mirror_list_file_name",
    "simtel_telescope_config_file_name",
    "validate_array_layout_name",
    "validate_model_version_name",
    "validate_site_name",
    "validate_telescope_id_name",
    "validate_telescope_name",
]

# Telescopes and other array elements
array_element_names = {
    "LSTN": {
        "site": "North",
        "observatory": "CTAO",
    },
    "MSTN": {
        "site": "North",
        "observatory": "CTAO",
    },
    "LSTS": {
        "site": "South",
        "observatory": "CTAO",
    },
    "MSTS": {
        "site": "South",
        "observatory": "CTAO",
    },
    "SSTS": {
        "site": "South",
        "observatory": "CTAO",
    },
    "SCTS": {
        "site": "South",
        "observatory": "CTAO",
    },
    "ILLN": {
        "site": "North",
        "observatory": "CTAO",
    },
    "MAGIC": {
        "site": "North",
        "observatory": "MAGIC",
    },
    "VERITAS": {
        "site": "North",
        "observatory": "VERITAS",
    },
    "HESS": {
        "site": "South",
        "observatory": "HESS",
    },
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
    "2024-02-01": [""],
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

# simulation_model parameter naming to DB parameter naming mapping
# TODO - probably not necessary after updates to the database
# simtel: True if alternative "name" is used in simtools (e.g., ref_lat)
#         and in the model database.
site_parameters = {
    # Note inconsistency between old and new model
    # altitude was the corsika observation level in the old model
    "reference_point_altitude": {"db_name": "altitude", "simtel": True},
    "reference_point_longitude": {"db_name": "ref_long", "simtel": False},
    "reference_point_latitude": {"db_name": "ref_lat", "simtel": False},
    "reference_point_utm_north": {"db_name": "reference_point_utm_north", "simtel": False},
    "reference_point_utm_east": {"db_name": "reference_point_utm_east", "simtel": False},
    # Note naming inconsistency between old and new model
    # altitude was the corsika observation level in the old model
    "corsika_observation_level": {"db_name": "altitude", "simtel": True},
    "epsg_code": {"db_name": "epsg_code", "simtel": False},
    "magnetic_field": {"db_name": "magnetic_field", "simtel": False},
    "atmospheric_profile": {"db_name": "atmospheric_profile", "simtel": False},
    "atmospheric_transmission": {"db_name": "atmospheric_transmission", "simtel": True},
    "array_coordinates": {"db_name": "array_coordinates", "simtel": False},
}

telescope_parameters = {
    "pixel_shape": {"db_name": "pixel_shape", "simtel": False},
    "pixel_diameter": {"db_name": "pixel_diameter", "simtel": False},
    "lightguide_efficiency_angle_file": {
        "db_name": "lightguide_efficiency_angle_file",
        "simtel": False,
    },
    "lightguide_efficiency_wavelength_file": {
        "db_name": "lightguide_efficiency_wavelength_file",
        "simtel": False,
    },
    "mirror_panel_shape": {"db_name": "mirror_panel_shape", "simtel": False},
    "mirror_panel_diameter": {"db_name": "mirror_panel_diameter", "simtel": False},
    "telescope_axis_height": {"db_name": "telescope_axis_height", "simtel": False},
    "telescope_sphere_radius": {"db_name": "telescope_sphere_radius", "simtel": False},
}


def validate_telescope_id_name(name):
    """
    Validate telescope ID. Allowed IDs are
    - DESIGN (for design telescopes or testing)
    - telescope ID (e.g., 1, 5, 15)
    - TEST (for testing)

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
    if name.upper() in ("DESIGN", "TEST"):
        return str(name).upper()

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
    Validate telescope name (e.g., MSTN-Design, MSTN-01).

    Parameters
    ----------
    name: str
        Telescope name.

    Returns
    -------
    str
        Validated name.
    """
    _tel_type, _tel_id = name.split("-")
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


def simtel_telescope_config_file_name(
    site, telescope_model_name, model_version, label, extra_label
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
    name = f"CTA-{site}-{telescope_model_name}-{model_version}"
    name += f"_{label}" if label is not None else ""
    name += f"_{extra_label}" if extra_label is not None else ""
    name += ".cfg"
    return name


def simtel_array_config_file_name(array_name, site, model_version, label):
    """
    sim_telarray config file name for an array.

    Parameters
    ----------
    array_name: str
        Prod5, ...
    site: str
        South or North.
    model_version: str
        Version of the model.
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    name = f"CTA-{array_name}-{site}-{model_version}"
    name += f"_{label}" if label is not None else ""
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


def ray_tracing_file_name(
    site,
    telescope_model_name,
    source_distance,
    zenith_angle,
    off_axis_angle,
    mirror_number,
    label,
    base,
):
    """
    File name for files required at the RayTracing class.

    Parameters
    ----------
    site: str
        South or North.
    telescope_model_name: str
        LST-1, MST-FlashCam, ...
    source_distance: float
        Source distance (km).
    zenith_angle: float
        Zenith angle (deg).
    off_axis_angle: float
        Off-axis angle (deg).
    mirror_number: int
        Mirror number. None if not single mirror case.
    label: str
        Instance label.
    base: str
        Photons, stars or log.

    Returns
    -------
    str
        File name.
    """
    name = (
        f"{base}-{site}-{telescope_model_name}-d{source_distance:.1f}"
        f"-za{zenith_angle:.1f}-off{off_axis_angle:.3f}"
    )
    name += f"_mirror{mirror_number}" if mirror_number is not None else ""
    name += f"_{label}" if label is not None else ""
    name += ".log" if base == "log" else ".lis"
    return name


def ray_tracing_results_file_name(site, telescope_model_name, source_distance, zenith_angle, label):
    """
    Ray tracing results file name.

    Parameters
    ----------
    site: str
        South or North.
    telescope_model_name: str
        LST-1, MST-FlashCam, ...
    source_distance: float
        Source distance (km).
    zenith_angle: float
        Zenith angle (deg).
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    name = f"ray-tracing-{site}-{telescope_model_name}-d{source_distance:.1f}-za{zenith_angle:.1f}"
    name += f"_{label}" if label is not None else ""
    name += ".ecsv"
    return name


def ray_tracing_plot_file_name(
    key, site, telescope_model_name, source_distance, zenith_angle, label
):
    """
    Ray tracing plot file name.

    Parameters
    ----------
    key: str
        Quantity to be plotted (d80_cm, d80_deg, eff_area or eff_flen)
    site: str
        South or North.
    telescope_model_name: str
        LST-1, MST-FlashCam, ...
    source_distance: float
        Source distance (km).
    zenith_angle: float
        Zenith angle (deg).
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    name = (
        f"ray-tracing-{site}-{telescope_model_name}-{key}-"
        f"d{source_distance:.1f}-za{zenith_angle:.1f}"
    )
    name += f"_{label}" if label is not None else ""
    name += ".pdf"
    return name


def camera_efficiency_results_file_name(
    site, telescope_model_name, zenith_angle, azimuth_angle, label
):
    """
    Camera efficiency results file name.

    Parameters
    ----------
    site: str
        South or North.
    telescope_model_name: str
        LST-1, MST-FlashCam, ...
    zenith_angle: float
        Zenith angle (deg).
    azimuth_angle: float
        Azimuth angle (deg).
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    _label = f"_{label}" if label is not None else ""
    name = (
        f"camera-efficiency-table-{site}-{telescope_model_name}-"
        f"za{round(zenith_angle):03}deg_azm{round(azimuth_angle):03}deg"
        f"{_label}.ecsv"
    )
    return name


def camera_efficiency_simtel_file_name(
    site, telescope_model_name, zenith_angle, azimuth_angle, label
):
    """
    Camera efficiency simtel output file name.

    Parameters
    ----------
    site: str
        South or North.
    telescope_model_name: str
        LST-1, MST-FlashCam-D, ...
    zenith_angle: float
        Zenith angle (deg).
    azimuth_angle: float
        Azimuth angle (deg).
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    _label = f"_{label}" if label is not None else ""
    name = (
        f"camera-efficiency-{site}-{telescope_model_name}-"
        f"za{round(zenith_angle):03}deg_azm{round(azimuth_angle):03}deg"
        f"{_label}.dat"
    )
    return name


def camera_efficiency_log_file_name(site, telescope_model_name, zenith_angle, azimuth_angle, label):
    """
    Camera efficiency log file name.

    Parameters
    ----------
    site: str
        South or North.
    telescope_model_name: str
        LST-1, MST-FlashCam-D, ...
    zenith_angle: float
        Zenith angle (deg).
    azimuth_angle: float
        Azimuth angle (deg).
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    _label = f"_{label}" if label is not None else ""
    name = (
        f"camera-efficiency-{site}-{telescope_model_name}"
        f"-za{round(zenith_angle):03}deg_azm{round(azimuth_angle):03}deg"
        f"{_label}.log"
    )
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
