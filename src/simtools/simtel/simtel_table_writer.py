"""Writer for sim_telarray table data files."""

import logging
from pathlib import Path

import numpy as np
from astropy.table import Table

from simtools.data_model import row_table_utils
from simtools.data_model.mirror_segmentation import (
    write_mirror_segmentation as _write_mirror_segmentation,
)
from simtools.simtel.pulse_shapes import generate_pulse_from_rise_fall_times

logger = logging.getLogger(__name__)


def write_mirror_segmentation(records, output_path, parameter_name, schema_version):
    """Write validated mirror-segmentation records for sim_telarray."""
    return _write_mirror_segmentation(records, output_path, parameter_name, schema_version)


def write_camera_configuration(configuration, output_path):
    """Write validated camera components in sim_telarray camera syntax.

    ``configuration`` is a mapping containing ``rotate``, ``pixel_types``,
    ``pixels`` and optional ``triggers``/``trigger_members`` sequences. The
    function deliberately accepts plain mappings so model-repository values
    can be passed without an intermediate bespoke class.
    """
    output_path = Path(output_path)
    _validate_camera_configuration(configuration)
    if any(part == ".." for part in output_path.parts):
        raise ValueError(f"Unsafe camera configuration path: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = _camera_configuration_lines(configuration)
    output_path.write_text("".join(lines), encoding="utf-8")
    return output_path.name


def _camera_configuration_lines(configuration):
    """Build serialized lines for a camera configuration."""
    rotate = float(configuration.get("rotate", 0.0))
    pixel_types = configuration.get("pixel_types", [])
    pixels = configuration.get("pixels", [])
    triggers = configuration.get("triggers", configuration.get("trigger_groups", []))
    members = configuration.get("trigger_members", [])
    members_by_group = {}
    for member in members:
        members_by_group.setdefault(member["group_id"], []).append(member)
    lines = ["# Generated from camera model parameters\n", f"Rotate {rotate:.12g}\n"]
    lines.extend(_pixel_type_lines(pixel_types))
    lines.extend(_pixel_lines(pixels))
    lines.extend(_trigger_lines(triggers, members_by_group))
    return lines


def _pixel_type_lines(pixel_types):
    """Build PixType lines."""
    lines = []
    for item in pixel_types:
        angle_file = item.get("lightguide_angle_file")
        wavelength_file = item.get("lightguide_wavelength_file")
        if not angle_file and item.get("funnel_transparency") is None:
            raise ValueError("Camera pixel type has no resolved lightguide or transparency")
        fields = [
            "PixType",
            str(item["type_id"]),
            str(item["pmt_type"]),
            str(item["cathode_shape"]),
            str(item["cathode_diameter_cm"]),
            str(item["funnel_shape"]),
            str(item["funnel_diameter_cm"]),
            str(item["funnel_depth_cm"]),
        ]
        if angle_file:
            fields.append(f'"{_safe_basename(angle_file, "lightguide angle")}"')
        else:
            fields.extend((str(item["funnel_transparency"]), str(item["funnel_wall_reflectivity"])))
        if wavelength_file:
            fields.append(f'"{_safe_basename(wavelength_file, "lightguide wavelength")}"')
        lines.append(" ".join(fields) + "\n")
    return lines


def _pixel_lines(pixels):
    """Build Pixel lines."""
    lines = []
    for pixel in pixels:
        values = [
            pixel["pixel_id"],
            pixel["type_id"],
            pixel["x_cm"],
            pixel["y_cm"],
            pixel["module"],
            pixel["board"],
            pixel["channel"],
            _module_id(pixel["module_id"]),
            int(bool(pixel["enabled"])),
            pixel["relative_qe"],
            pixel["relative_gain"],
            pixel["z_offset_cm"],
            pixel["rotation_deg"],
            pixel["normal_x"],
            pixel["normal_y"],
        ]
        lines.append("Pixel " + " ".join(str(value) for value in values) + "\n")
    return lines


def _trigger_lines(triggers, members_by_group):
    """Build trigger lines."""
    return [
        _trigger_line(trigger, index, members_by_group) for index, trigger in enumerate(triggers)
    ]


def _trigger_line(trigger, index, members_by_group):
    """Build one trigger line."""
    keyword = {
        "majority": "MajorityTrigger",
        "analogsum": "AnalogSumTrigger",
        "digitalsum": "DigitalSumTrigger",
    }.get(trigger["kind"].lower())
    if keyword is None:
        raise ValueError(f"Unsupported camera trigger kind: {trigger['kind']}")
    multiplicity = (
        "*" if bool(trigger["use_default_multiplicity"]) else str(trigger["multiplicity"])
    )
    group_id = trigger.get("group_id", index)
    tokens = _trigger_member_tokens(members_by_group.get(group_id, []))
    return f"{keyword} {multiplicity} of {' '.join(tokens)}\n"


def _trigger_member_tokens(members):
    """Build sim_telarray tokens for normalized trigger members."""
    grouped_members = {}
    for member in members:
        grouped_members.setdefault(member["member_order"], []).append(member)
    return [_trigger_member_token(rows) for rows in grouped_members.values()]


def _trigger_member_token(member_rows):
    """Build one scalar or bracketed trigger member token."""
    member_rows.sort(key=lambda row: row["pixel_order"])
    first = member_rows[0]
    prefix = "+" if bool(first["required"]) else ""
    if len(member_rows) == 1:
        return prefix + str(first["pixel_id"])
    slaves = ",".join(str(row["pixel_id"]) for row in member_rows[1:])
    return f"{prefix}{first['pixel_id']}[{slaves}]"


def _validate_camera_configuration(configuration):
    """Validate camera component records before serializing them."""
    pixel_types = configuration.get("pixel_types", [])
    pixels = configuration.get("pixels", [])
    triggers = configuration.get("triggers", configuration.get("trigger_groups", []))
    members = configuration.get("trigger_members", [])
    if not pixel_types or not pixels:
        raise ValueError("Camera configuration requires pixel types and pixels")
    type_ids = _validate_pixel_types(pixel_types)
    _validate_pixels(pixels, type_ids)
    _validate_triggers(triggers, members, pixels)


def _validate_pixel_types(pixel_types):
    """Validate pixel types and return their IDs."""
    type_ids = [item.get("type_id") for item in pixel_types]
    if len(set(type_ids)) != len(type_ids):
        raise ValueError("Camera pixel type IDs must be unique")
    for item in pixel_types:
        if item.get("lightguide_angle_file"):
            _safe_basename(item["lightguide_angle_file"], "lightguide angle")
        elif (
            item.get("funnel_transparency") is None or item.get("funnel_wall_reflectivity") is None
        ):
            raise ValueError("Camera pixel type has no resolved lightguide or transparency")
    return type_ids


def _validate_pixels(pixels, type_ids):
    """Validate pixel ordering and foreign keys."""
    pixel_ids = [item.get("pixel_id") for item in pixels]
    if pixel_ids != list(range(len(pixels))):
        raise ValueError("Camera pixel IDs must be contiguous and ordered")
    if not any(bool(item.get("enabled")) for item in pixels):
        raise ValueError("Camera configuration must contain an enabled pixel")
    if any(item.get("type_id") not in type_ids for item in pixels):
        raise ValueError("Camera pixel references an unknown pixel type")


def _validate_triggers(triggers, members, pixels):
    """Validate trigger groups and their foreign keys."""
    pixel_ids = [item.get("pixel_id") for item in pixels]
    group_ids = [item.get("group_id") for item in triggers]
    if group_ids != list(range(len(triggers))):
        raise ValueError("Camera trigger group IDs must be contiguous and ordered")
    members_by_group = {}
    for member in members:
        members_by_group.setdefault(member.get("group_id"), []).append(member)
    if set(members_by_group) - set(group_ids):
        raise ValueError("Camera trigger member references an unknown group")
    for trigger in triggers:
        _validate_trigger(trigger, members_by_group.get(trigger["group_id"], []), pixel_ids)


def _validate_trigger(trigger, members, pixel_ids):
    """Validate one trigger group and its normalized member rows."""
    _validate_trigger_kind(trigger)
    use_default = bool(trigger.get("use_default_multiplicity"))
    multiplicity = trigger.get("multiplicity")
    _validate_trigger_multiplicity(use_default, multiplicity)
    if not members:
        raise ValueError(f"Camera trigger group has no members: {trigger['group_id']}")
    _validate_trigger_members(members, pixel_ids)


def _validate_trigger_kind(trigger):
    """Validate the trigger kind."""
    if trigger.get("kind", "").lower() not in {"majority", "analogsum", "digitalsum"}:
        raise ValueError(f"Unsupported camera trigger kind: {trigger.get('kind')}")


def _validate_trigger_multiplicity(use_default, multiplicity):
    """Validate default or explicit trigger multiplicity."""
    if use_default:
        if multiplicity not in (None, 0):
            raise ValueError("Default trigger multiplicity must not be positive")
        return
    if multiplicity is None or int(multiplicity) < 1:
        raise ValueError("Explicit trigger multiplicity must be positive")


def _validate_trigger_members(members, pixel_ids):
    """Validate normalized trigger member rows."""
    member_orders = sorted({member["member_order"] for member in members})
    if member_orders != list(range(len(member_orders))):
        raise ValueError("Camera trigger member orders must be contiguous")
    for member_order in member_orders:
        rows = sorted(
            (row for row in members if row["member_order"] == member_order),
            key=lambda row: row["pixel_order"],
        )
        if [row["pixel_order"] for row in rows] != list(range(len(rows))):
            raise ValueError("Camera trigger pixel orders must be contiguous")
        if rows[0]["required"] and any(row["required"] for row in rows[1:]):
            raise ValueError("Only the first pixel of a trigger member may be required")
        if any(row["pixel_id"] not in pixel_ids for row in rows):
            raise ValueError("Camera trigger contains an unknown pixel ID")


def _safe_basename(value, label):
    """Return a safe generated dependency basename."""
    path = Path(value)
    if path.name != str(value) or path.name in {"", ".", ".."}:
        raise ValueError(f"Unsafe {label} filename: {value}")
    return path.name


def _module_id(value):
    """Format a module ID as a safe hexadecimal sim_telarray token."""
    try:
        integer = int(str(value), 0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid camera module ID: {value}") from exc
    if integer < 0:
        raise ValueError(f"Invalid camera module ID: {value}")
    return f"0x{integer:x}"


def write_simtel_table(table_or_parameter, value_or_dest, dest_dir=None, telescope_name=None):
    """Write a table parameter to a space-separated ASCII file for sim_telarray.

    Parameters
    ----------
    parameter_name : str
        Parameter name, used as filename prefix.
    value : dict
        Table data with keys ``columns`` (list of str) and ``rows`` (list of lists).
    dest_dir : str or Path
        Directory to write the file into.
    telescope_name : str
        Telescope name, used as filename suffix.

    Returns
    -------
    str
        Basename of the written file (``{parameter_name}-{telescope_name}.dat``).

    Raises
    ------
    ValueError
        If ``value`` does not contain ``columns`` and ``rows`` keys.
    """
    if isinstance(table_or_parameter, Table):
        return _write_ecsv_table(table_or_parameter, value_or_dest)

    parameter_name = table_or_parameter
    value = value_or_dest
    if not isinstance(value, dict) or "columns" not in value or "rows" not in value:
        raise ValueError(
            f"Table value for '{parameter_name}' must be a dict with 'columns' and 'rows' keys, "
            f"got {type(value).__name__}."
        )

    row_table_utils.validate_row_table_structure(parameter_name, value, require_column_units=False)

    file_name = f"{parameter_name}-{telescope_name}.dat"
    file_path = Path(dest_dir) / file_name
    logger.debug(f"Writing sim_telarray table file {file_path}")

    with open(file_path, "w", encoding="utf-8") as fh:
        fh.write(f"# {' '.join(value['columns'])}\n")
        for row in value["rows"]:
            fh.write(" ".join(str(v) for v in row) + "\n")

    return file_name


def _write_ecsv_table(table, dest_dir):
    """Write a validated ECSV table in its original sim_telarray representation."""
    output_name = table.meta.get("simtelarray_original_file_name")
    if not output_name:
        raise ValueError("ECSV table metadata must define simtelarray_original_file_name")
    output_path = Path(dest_dir) / Path(output_name).name
    if Path(output_name).name != output_name:
        raise ValueError(f"Unsafe sim_telarray output filename: {output_name}")

    format_name = table.meta.get("simtelarray_table_format", "plain")
    writers = {
        "plain": _write_plain_table,
        "pulse": _write_plain_table,
        "mirror_list": _write_plain_table,
        "rpol_matrix": _write_rpol_table,
        "atmospheric_transmission": _write_atmospheric_transmission,
    }
    try:
        writers[format_name](table, output_path)
    except KeyError as exc:
        raise ValueError(f"Unknown sim_telarray table format: {format_name}") from exc
    return output_path.name


def _table_comments(table):
    """Return source comments as lines suitable for an ASCII table."""
    comments = table.meta.get("original_comments", [])
    if isinstance(comments, str):
        comments = comments.splitlines()
    return [f"# {comment}" if comment else "#" for comment in comments]


def _row_values(table, row):
    """Return serializable scalar values from an Astropy row."""
    values = []
    for name in table.colnames:
        value = row[name]
        values.append(getattr(value, "value", value))
    return values


def _raw_values(values):
    """Return plain scalar values from an Astropy column or iterable."""
    return [getattr(value, "value", value) for value in values]


def _write_plain_table(table, output_path):
    """Write comments and one whitespace-separated row per table row."""
    with output_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(_table_comments(table)))
        if table.meta.get("original_comments"):
            file.write("\n")
        for row in table:
            file.write(" ".join(str(value) for value in _row_values(table, row)) + "\n")


def _write_rpol_table(table, output_path):
    """Write a tidy wavelength/angle table in sim_telarray RPOL format."""
    angle_name = "angle" if "angle" in table.colnames else "incidence_angle"
    independent_name = "wavelength"
    dependent = table.meta.get("simtelarray_value_column")
    if dependent is None:
        dependent = next(
            (
                name
                for name in ("reflectivity", "transmission", "efficiency")
                if name in table.colnames
            ),
            None,
        )
    if dependent is None:
        raise ValueError("RPOL ECSV table must define simtelarray_value_column")
    if angle_name not in table.colnames or independent_name not in table.colnames:
        raise ValueError("RPOL ECSV table must contain wavelength and angle columns")
    angles = list(dict.fromkeys(_raw_values(table[angle_name])))
    values = {}
    for row in table:
        key = (
            getattr(row[independent_name], "value", row[independent_name]),
            getattr(row[angle_name], "value", row[angle_name]),
        )
        if key in values:
            raise ValueError("RPOL ECSV table must contain one value per wavelength and angle")
        values[key] = getattr(row[dependent], "value", row[dependent])
    comments = [
        line
        for line in _table_comments(table)
        if not any(token in line for token in ("@RPOL@", "ANGLE=", "H1=", "H2="))
    ]
    with output_path.open("w", encoding="utf-8") as file:
        for line in comments:
            file.write(f"{line}\n")
        file.write("#@RPOL@[ANGLE=] 2\n")
        file.write("ANGLE= " + " ".join(str(angle) for angle in angles) + "\n")
        wavelengths = list(dict.fromkeys(_raw_values(table[independent_name])))
        for wavelength in wavelengths:
            selection = []
            for angle in angles:
                try:
                    selection.append(values[(wavelength, angle)])
                except KeyError as exc:
                    raise ValueError(
                        "RPOL ECSV table must contain one value per wavelength and angle"
                    ) from exc
            file.write(" ".join([str(wavelength), *(str(value) for value in selection)]) + "\n")


def _write_atmospheric_transmission(table, output_path):
    """Write a tidy atmospheric transmission table in sim_telarray matrix format."""
    altitude_name = "altitude"
    dependent = "extinction"
    altitudes = list(dict.fromkeys(_raw_values(table[altitude_name])))
    with output_path.open("w", encoding="utf-8") as file:
        for line in _table_comments(table):
            file.write(f"{line}\n")
        file.write("# H1= " + " ".join(str(value) for value in altitudes) + "\n")
        for wavelength in dict.fromkeys(_raw_values(table["wavelength"])):
            values = [
                getattr(row[dependent], "value", row[dependent])
                for row in table
                if getattr(row["wavelength"], "value", row["wavelength"]) == wavelength
            ]
            file.write(" ".join([str(wavelength), *(str(value) for value in values)]) + "\n")


def write_light_pulse_table_gauss_exp_conv(
    file_path,
    width_ns,
    exp_decay_ns,
    fadc_sum_bins,
    dt_ns=0.1,
    rise_range=(0.1, 0.9),
    fall_range=(0.9, 0.1),
    time_margin_ns=10.0,
):
    """Write a pulse table for a Gaussian convolved with a causal exponential.

    Parameters
    ----------
    file_path : str or Path
        Destination path of the ASCII pulse table. Parent directory must exist.
    width_ns : float
        Rise time in ns between the fractional levels defined by ``rise_range``.
    exp_decay_ns : float
        Fall time in ns between the fractional levels defined by ``fall_range``.
    fadc_sum_bins : int
        FADC integration window length in bins, used to set the time range.
    dt_ns : float, optional
        Time sampling step in ns.
    rise_range : tuple[float, float], optional
        Fractional amplitude bounds (low, high) for rise-time definition.
    fall_range : tuple[float, float], optional
        Fractional amplitude bounds (high, low) for fall-time definition.
    time_margin_ns : float, optional
        Extra margin in ns added to both ends of the time window.

    Returns
    -------
    Path
        Path to the created pulse table file.

    Raises
    ------
    ValueError
        If ``width_ns`` or ``exp_decay_ns`` is None.
    """
    if width_ns is None or exp_decay_ns is None:
        raise ValueError("width_ns (rise 10-90) and exp_decay_ns (fall 90-10) are required")
    logger.info(
        "Generating pulse-shape table with "
        f"rise{int(rise_range[0] * 100)}-{int(rise_range[1] * 100)}={width_ns} ns, "
        f"fall{int(fall_range[0] * 100)}-{int(fall_range[1] * 100)}={exp_decay_ns} ns, "
        f"dt={dt_ns} ns"
    )
    width = float(fadc_sum_bins)
    t_start_ns = -abs(time_margin_ns + width)
    t_stop_ns = +abs(time_margin_ns + width)
    t, y = generate_pulse_from_rise_fall_times(
        width_ns,
        exp_decay_ns,
        dt_ns=dt_ns,
        rise_range=rise_range,
        fall_range=fall_range,
        t_start_ns=t_start_ns,
        t_stop_ns=t_stop_ns,
        center_on_peak=True,
    )

    return write_ascii_pulse_table(file_path, t, y)


def write_angular_distribution_table_lambertian(
    file_path,
    max_angle_deg,
    n_samples=100,
):
    """Write a Lambertian angular distribution table (intensity ~ cos(angle)).

    Parameters
    ----------
    file_path : str or Path
        Destination path of the ASCII table. Parent directory must exist.
    max_angle_deg : float
        Upper bound of the angular range in degrees.
    n_samples : int, optional
        Number of equally spaced samples from 0 to ``max_angle_deg``.

    Returns
    -------
    Path
        Path to the created angular distribution table.
    """
    logger.info(
        f"Generating Lambertian angular distribution table up to {max_angle_deg} deg "
        f"with {n_samples} samples"
    )
    angles = np.linspace(0.0, float(max_angle_deg), int(n_samples), dtype=float)
    intensities = np.cos(np.deg2rad(angles))
    intensities[intensities < 0] = 0.0
    if intensities.max() > 0:
        intensities /= intensities.max()

    return write_ascii_angle_distribution_table(file_path, angles, intensities)


def write_ascii_pulse_table(file_path, t, y):
    """Write a two-column (time, amplitude) ASCII pulse table.

    Parameters
    ----------
    file_path : str or Path
        Destination path.
    t : array-like
        Time values in ns.
    y : array-like
        Amplitude values.

    Returns
    -------
    Path
        Path to the written file.
    """
    with open(file_path, "w", encoding="utf-8") as fh:
        fh.write("# time[ns] amplitude\n")
        for ti, yi in zip(t, y):
            fh.write(f"{ti:.6f} {yi:.8f}\n")
    return Path(file_path)


def write_ascii_angle_distribution_table(file_path, angles, intensities):
    """Write a two-column (angle, relative intensity) ASCII angular distribution table.

    Parameters
    ----------
    file_path : str or Path
        Destination path.
    angles : array-like
        Angle values in degrees.
    intensities : array-like
        Relative intensity values.

    Returns
    -------
    Path
        Path to the written file.
    """
    with open(file_path, "w", encoding="utf-8") as fh:
        fh.write("# angle[deg] relative_intensity\n")
        for a, i in zip(angles, intensities):
            fh.write(f"{a:.6f} {i:.8f}\n")
    return Path(file_path)
