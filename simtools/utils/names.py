import logging
import re

_logger = logging.getLogger(__name__)

__all__ = [
    "array_element_id_from_telescope_model_name",
    "camera_efficiency_log_file_name",
    "camera_efficiency_results_file_name",
    "camera_efficiency_simtel_file_name",
    "convert_telescope_model_name_to_yaml_name",
    "get_site_from_telescope_name",
    "get_telescope_class",
    "layout_telescope_list_file_name",
    "ray_tracing_file_name",
    "ray_tracing_plot_file_name",
    "ray_tracing_results_file_name",
    "sanitize_name",
    "simtel_array_config_file_name",
    "simtel_single_mirror_list_file_name",
    "simtel_telescope_config_file_name",
    "get_telescope_name_db",
    "split_telescope_model_name",
    "telescope_model_name_from_array_element_id",
    "translate_simtools_to_corsika",
    "validate_array_layout_name",
    "validate_model_version_name",
    "validate_site_name",
    "validate_sub_system_name",
    "validate_telescope_id_name",
    "validate_telescope_model_name",
    "validate_telescope_name_db",
]

lst = "LST"
mst = "MST"
sct = "SCT"
sst = "SST"
hess = "HESS"
magic = "MAGIC"
veritas = "VERITAS"

all_telescope_class_names = {
    lst: ["lst"],
    mst: ["mst"],
    sct: ["sct"],
    sst: ["sst"],
    hess: ["hess"],
    magic: ["magic"],
    veritas: ["veritas"],
}

all_subsystem_names = {
    "SST": ["sst", "ssts", "SSTS"],
    "ASTRI": ["astri"],
    "GCT": ["gct", "gct-s"],
    "1M": ["1m"],
    "FlashCam": ["flashcam", "flash-cam", "msts", "MSTS"],
    "NectarCam": ["nectarcam", "nectar-cam", "mstn", "MSTN"],
    "SCT": ["sct", "scts", "SCTS"],
    "LST": ["lst", "lsts", "LSTS", "lstn", "LSTN"],
    "Camera": ["Camera", "camera"],
    "Structure": ["Structure", "structure"],
    "MAGIC": ["magic", "MAGIC"],
    "VERITAS": ["veritas", "VERITAS"],
    "HESS": ["hess", "HESS"],
}

all_site_names = {
    "South": ["paranal", "south", "cta-south", "ctao-south", "s"],
    "North": ["lapalma", "north", "cta-north", "ctao-north", "n"],
}

all_model_version_names = {
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
    "prod4-prototype": [""],
    "default": [],
    "Released": [],
    "Latest": [],
}

all_array_layout_names = {
    "4LST": ["4-lst", "4lst"],
    "1LST": ["1-lst", "1lst"],
    "4MST": ["4-mst", "4mst"],
    "1MST": ["1-mst", "mst"],
    "4SST": ["4-sst", "4sst"],
    "1SST": ["1-sst", "sst"],
    "Prod5": ["prod5", "p5"],
    "TestLayout": ["test-layout"],
}

# array elements as defined by CTAO
# (includes for now only telescopes;
# can be extended to other elements)
all_array_element_id_names = {
    "lstn": ["LSTN", "lstn"],
    "mstn": ["MSTN", "mstn"],
    "lsts": ["LSTS", "lsts"],
    "msts": ["MSTS", "msts"],
    "ssts": ["SSTS", "ssts"],
    "scts": ["SCTS", "scts"],
}


def validate_sub_system_name(name):
    """
    Validate a sub system name (optics structure or camera).

    Parameters
    ----------
    name: str
        Name of the subsystem.

    Returns
    -------
    str
        Validated name.
    """
    return _validate_name(name, all_subsystem_names)


def validate_telescope_class_name(name):
    """
    Validate a telescope class name (LST, MST, SST, ...).

    Parameters
    ----------
    name: str
        Telescope class name.

    Returns
    -------
    str
        Validated name.
    """
    return _validate_name(name, all_telescope_class_names)


def validate_telescope_id_name(name):
    """
    Validate a telescope ID name

    Valid names e.g.,
    - D
    - telescope ID (e.g., 1, 5, 15)

    Parameters
    ----------
    name: str or int
        Telescope ID name.

    Returns
    -------
    str
        Validated name.

    Raises
    ------
    ValueError
        If name is not valid.
    """

    if isinstance(name, int) or name.upper() in ("D", "D234", "TEST") or name.isdigit():
        return str(name).upper()

    msg = f"Invalid telescope ID name {name}"
    _logger.error(msg)
    raise ValueError(msg)


def validate_model_version_name(name):
    """
    Validate a model version name.

    Parameters
    ----------
    name: str
        Model version name.

    Returns
    -------
    str
        Validated name.
    """
    return _validate_name(name, all_model_version_names)


def validate_site_name(name):
    """
    Validate a site name.

    Parameters
    ----------
    name: str
        Site name.

    Returns
    -------
    str
        Validated name.
    """
    return _validate_name(name, all_site_names)


def validate_array_layout_name(name):
    """
    Validate a array layout name.

    Parameters
    ----------
    name: str
        Layout array name.

    Returns
    -------
    str
        Validated name.
    """
    return _validate_name(name, all_array_layout_names)


def _validate_name(name, all_names):
    """
    Validate a name given the all_names options. For each key in all_names, a list of options is \
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

    if not _is_valid_name(name, all_names):
        msg = f"Invalid name {name}"
        raise ValueError(msg)
    for main_name, list_of_names in all_names.items():
        if name.lower() in list_of_names + [main_name.lower()]:
            if name != main_name:
                _logger.debug(f"Correcting name {name} -> {main_name}")
            return main_name
    return None


def _is_valid_name(name, all_names):
    """
    Check if name is valid.

    Parameters
    ----------
    name:  str
        Name to validated.
    all_names: dict
        Dictionary with valid names.

    Returns
    -------
    bool
        True if name is valid. Otherwise, false.
    """

    if not isinstance(name, str):
        return False
    for main_name in all_names.keys():
        if name.lower() in all_names[main_name] + [main_name.lower()]:
            return True
    return False


def validate_telescope_model_name(name):
    """
    Validate a telescope model name.

    Parameters
    ----------
    name: str
        Telescope model name.

    Returns
    -------
    str
        Validated name.
    """

    # e.g, MSTN or MSTN-01
    try:
        return _validate_name(name, all_array_element_id_names)
    except ValueError:
        pass

    # e.g., MST-FlashCam or MST-FlashCam-01
    tel_class, tel_type, tel_id = split_telescope_model_name(name)
    _telescope_name = tel_class
    if tel_class != tel_type and len(tel_type) > 0:
        _telescope_name += "-" + tel_type
    if len(tel_id) > 0:
        _telescope_name += "-" + tel_id
    return _telescope_name


def split_telescope_model_name(name):
    """
    Split a telescope name into class, type, telescope id.
    Allow various inputs, e.g., MST-FlashCam-01, MST-FlashCam,
    MSTN-01, MSTN, MST-1

    Parameters
    ----------
    name: str
        Telescope name.

    Returns
    -------
    str, str, str
       class (LST, MST, SST ...), type (any complement), telescope id
    """
    name_parts = name.split("-")
    # e.g., MSTN or MSTN-01
    try:
        _is_valid_name(name_parts[0], all_array_element_id_names)
        return (
            validate_telescope_class_name(name_parts[0][0:3]),
            validate_sub_system_name(name_parts[0]),
            validate_telescope_id_name(name_parts[1]) if len(name_parts) > 1 else "",
        )
    except ValueError:
        pass

    # e.g., MST-FlashCam or MST-FlashCam-01, MST-1
    # (note that this complicated, as LST-1 is a valid
    # name without explicit subsystem given in name)
    tel_class = name_parts[0]
    try:
        tel_id = validate_telescope_id_name(name_parts[-1])
        _tmp_tel_type = name_parts[1:-1]
    except ValueError:
        tel_id = ""
        _tmp_tel_type = name_parts[1:]
    if len(_tmp_tel_type) > 0:
        tel_type = "-".join(_tmp_tel_type)
    else:
        try:
            tel_type = validate_sub_system_name(tel_class)
        except ValueError:
            tel_type = ""

    return (
        validate_telescope_class_name(tel_class),
        validate_sub_system_name(tel_type) if len(tel_type) > 0 else "",
        validate_telescope_id_name(tel_id) if len(tel_id) > 0 else "",
    )


def get_site_from_telescope_name(name):
    """
    Get site name (South or North) from the (validated) telescope name.

    Parameters
    ----------
    name: str
        Telescope name.

    Returns
    -------
    str
        Site name (South or North).
    """
    # e.g, MSTN or MSTN-01 (tested by get_telescope_class)
    try:
        get_telescope_class(name)
        return validate_site_name(name[3])
    except ValueError:
        pass
    # e.g., South-MST-FlashCam
    return validate_site_name(name.split("-")[0])


def validate_telescope_name_db(name):
    """
    Validate a telescope DB name.
    Examples are North-LST-1, North-MST-NectarCam-D, or South-SST-Structure-D.

    Parameters
    ----------
    name: str

    Returns
    -------
    str
        Validated name.
    """
    site = get_site_from_telescope_name(name)
    tel_model_name = "-".join(name.split("-")[1:])

    return f"{validate_site_name(site)}-{validate_telescope_model_name(tel_model_name)}"


def convert_telescope_model_name_to_yaml_name(name):
    """
    Get telescope name following the old convention (yaml files) from the current telescope name.

    Parameters
    ----------
    name: str
        Telescope model name.

    Returns
    -------
    str
        Telescope name (old convention).

    Raises
    ------
    ValueError
        if name is not valid.
    """
    tel_class, tel_type, tel_id = split_telescope_model_name(name)
    new_name = tel_class if tel_class == tel_type else tel_class + "-" + tel_type
    new_name = new_name + "-" + tel_id if len(tel_id) > 0 else new_name
    old_names = {
        "SST-D": "SST",
        "SST-1M": "SST-1M",
        "SST-ASTRI": "SST-2M-ASTRI",
        "SST-GCT": "SST-2M-GCT-S",
        "MST-FlashCam-D": "MST-FlashCam",
        "MST-NectarCam-D": "MST-NectarCam",
        "SCT-D": "SCT",
        "LST-D234": "LST",
        "LST-1": "LST",
    }

    if new_name not in old_names:
        raise ValueError(f"Telescope name {name}/{new_name} could not be converted to yml names")

    return old_names[new_name]


def array_element_id_from_telescope_model_name(site, telescope_model_name):
    """
    Array element ID (CTAO convention) from telescope model name.
    This returns e.g., "LSTN" for any LST telescope in the North site.
    If a telescope number is given, it adds it as e.g., "LSTN-01".

    Parameters
    ----------
    site: str
        Observatory site (e.g., South or North)
    telescope_class_name: str
        Name of the telescope class (e.g. LST-1, LST-D234, MST-FlashCam-D, ...)

    Returns
    -------
    str
        Array element ID (CTAO style).

    """

    _class, _type, _tel_id = split_telescope_model_name(telescope_model_name)
    _id = _class.upper() + site[0].upper()
    if _tel_id.isdigit():
        _id += f"-{int(_tel_id):02d}"
    return _id


def telescope_model_name_from_array_element_id(
    array_element_id, sub_system_name="structure", available_telescopes=None
):
    """
    Telescope model name from array element ID (CTAO convention).
    Does not include the site in the returned name (e.g., returns
    South-MST-FlashCam-1 for MSTS-01; this method is quite finetuned).

    Parameters
    ----------
    array_element_id: str
        Array element ID (CTAO convention).
    available_telescopes: list
        List of available telescopes.

    Returns
    -------
    str
        Telescope model name.
    """

    name_parts = array_element_id.split("-")
    try:
        _class = name_parts[0][0:3]
        _site = validate_site_name(name_parts[0][3])
    except IndexError as exc:
        _logger.error("Invalid array element ID %s", array_element_id)
        raise exc
    try:
        _id = int(name_parts[1])
    except (ValueError, IndexError):
        _id = "D"

    if _class in ("LST", "SCT"):
        sub_system_name = None
    elif _class == "MST" and sub_system_name.lower() == "camera":
        sub_system_name = "NectarCam" if _site == "North" else "FlashCam"

    _simtools_name = get_telescope_name_db(
        site=_site,
        telescope_class_name=_class,
        sub_system_name=sub_system_name,
        telescope_id_name=_id,
    )
    if available_telescopes is not None and _simtools_name not in available_telescopes:
        _logger.debug("Telescope %s not available", _simtools_name)
        _simtools_name = get_telescope_name_db(
            site=_site,
            telescope_class_name=_class,
            sub_system_name=sub_system_name,
            telescope_id_name="D234" if _site == "North" and _class == "LST" else "D",
        )
    return _simtools_name.split("-", 1)[1]


def get_telescope_name_db(site, telescope_class_name, sub_system_name, telescope_id_name):
    """
    Instrument name following simtools naming convention
    Examples are North-LST-1, North-MST-NectarCam-D, or South-SST-Structure-D.

    Parameters
    ----------
    site: str
        South or North.
    telescope_class_name: str
        LST, MST, ...
    sub_system_name: str or None
        FlashCam, NectarCam, Structure
    telescope_id_name: str
        telescope ID (e.g., D, numerical value)

    Returns
    -------
    instrument: name: str
        Instrument name.
    """

    return (
        validate_site_name(site)
        + "-"
        + _validate_name(telescope_class_name, all_telescope_class_names)
        + ("" if sub_system_name is None else "-" + validate_sub_system_name(sub_system_name))
        + "-"
        + validate_telescope_id_name(telescope_id_name)
    )


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


def get_telescope_class(telescope_name):
    """
    Guess telescope class from name, e.g. "LST", "MST", ...

    Parameters
    ----------
    telescope_name: str
        Telescope name

    Returns
    -------
    str
        Telescope type.
    """

    _tel_class, _, _ = split_telescope_model_name(telescope_name)
    return _tel_class


def translate_simtools_to_corsika(simtools_par):
    """
    Translate the name of a simtools parameter to the name used in CORSIKA.

    TODO - this will go with the new simulation model

    Parameters
    ----------
    simtools_par: str
        Name of the simtools parameter to be translated.
    """

    corsika_to_simtools_names = {"OBSLEV": "corsika_obs_level"}

    simtools_to_corsika_names = {
        new_key: new_value for new_value, new_key in corsika_to_simtools_names.items()
    }
    try:
        return simtools_to_corsika_names[simtools_par]
    except KeyError:
        msg = f"Translation not found. We will proceed with the original parameter name:\
            {simtools_par}."
        _logger.debug(msg)
        return simtools_par


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
