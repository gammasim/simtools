"""Writer for sim_telarray table data files."""

import logging
from itertools import product
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import Table

from simtools.simtel.pulse_shapes import generate_pulse_from_rise_fall_times
from simtools.simtel.segmentation import write_mirror_segmentation as _write_mirror_segmentation
from simtools.simtel.table_serializers import (
    validate_simtel_serialization as _validate_contract_table,
)
from simtools.simtel.table_serializers import write_simtel_table as _write_contract_table

logger = logging.getLogger(__name__)


def write_mirror_segmentation(records, output_path, parameter_name, schema_version):
    """Write validated mirror-segmentation records for sim_telarray."""
    return _write_mirror_segmentation(records, output_path, parameter_name, schema_version)


def write_camera_file(camera_components, output_path):
    """Write validated camera components in sim_telarray camera syntax.

    ``camera_components`` is a mapping containing ``rotate``, ``pixel_types``,
    ``pixels`` and optional ``triggers``/``trigger_members`` sequences. The
    function deliberately accepts plain mappings so model-repository values
    can be passed without an intermediate bespoke class.
    """
    output_path = Path(output_path)
    _validate_camera_components(camera_components)
    if any(part == ".." for part in output_path.parts):
        raise ValueError(f"Unsafe camera configuration path: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = _camera_file_lines(camera_components)
    output_path.write_text("".join(lines), encoding="utf-8")
    return output_path.name


def _camera_file_lines(configuration):
    """Build serialized sim_telarray camera-file lines."""
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


def _validate_camera_components(configuration):
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


def write_simtel_table(
    table,
    dest_dir,
    table_format=None,
    output_name=None,
    contract=None,
):
    """Write a validated ECSV table using its sim_telarray contract.

    The contract controls positional column order, sorting, matrix axes, units,
    and numeric formatting. Source metadata and ECSV column order are never
    serialized.

    Parameters
    ----------
    table : astropy.table.Table
        Validated model table.
    dest_dir : str or pathlib.Path
        Directory receiving the generated simulator file.
    table_format : str, optional
        Serialization format. Prefer passing ``contract`` from the parameter
        schema; this argument is retained for direct adapter use.
    output_name : str, optional
        Deterministic output basename.
    contract : dict, optional
        Complete serialization contract.

    Returns
    -------
    str
        Basename of the generated file.

    Raises
    ------
    ValueError
        If the table does not satisfy the serialization contract.
    """
    if not isinstance(table, Table):
        raise TypeError(f"sim_telarray table writer requires an Astropy table, got {type(table)}")
    if table_format is not None:
        raise ValueError("table_format is schema-owned; pass the complete serialization contract")
    return _write_contract_table(table, dest_dir, contract, output_name)


def validate_simtel_serialization(table, contract):
    """Validate the positional and matrix invariants declared by a contract.

    Parameters
    ----------
    table : astropy.table.Table
        Table to validate.
    contract : dict
        Serialization contract from a model-parameter schema.

    Raises
    ------
    ValueError
        If columns, units, axes, or matrix cells violate the contract.
    """
    return _validate_contract_table(table, contract)


def _validate_contract_definition(contract):
    """Validate references between fields in a serialization contract."""
    columns = contract.get("columns", [])
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("sim_telarray serialization columns must be unique and non-empty")
    optional = set(contract.get("optional_columns", []))
    allowed = set(contract.get("allowed_columns", columns))
    declared = allowed | optional
    if not set(columns) <= declared:
        raise ValueError("sim_telarray serialization allowed columns must include columns")
    sort_keys = set(contract.get("row_sort_keys", contract.get("sort_keys", [])))
    if not sort_keys <= declared:
        raise ValueError("sim_telarray serialization sort keys must be declared columns")
    matrix_axes = contract.get("matrix_axes", [])
    if len(matrix_axes) not in (0, 2) or len(matrix_axes) != len(set(matrix_axes)):
        raise ValueError("sim_telarray serialization matrix_axes must contain two unique axes")
    if matrix_axes and not set(matrix_axes) <= declared:
        raise ValueError("sim_telarray serialization matrix axes must be declared columns")
    value_column = contract.get("value_column")
    if value_column is not None and value_column not in declared:
        raise ValueError("sim_telarray serialization value column must be declared")


def _validate_contract_columns(table, columns, contract):
    """Validate declared and present positional columns."""
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("sim_telarray serialization columns must be unique and non-empty")
    missing = sorted(set(columns) - set(table.colnames))
    if missing:
        raise ValueError(f"sim_telarray serialization is missing columns: {missing}")
    format_name = contract.get("table_format", "plain")
    if format_name in {"rpol_matrix", "atmospheric_transmission"}:
        axes = contract.get("matrix_axes", [])
        value_column = contract.get("value_column")
        if len(axes) != 2 or value_column is None:
            raise ValueError(f"{format_name} serialization requires two axes and a value column")
        if value_column not in table.colnames:
            raise ValueError(f"sim_telarray serialization is missing value column: {value_column}")
    allowed = set(contract.get("allowed_columns", columns))
    allowed.update(contract.get("optional_columns", []))
    unexpected = sorted(set(table.colnames) - allowed)
    if unexpected:
        raise ValueError(f"sim_telarray serialization has undeclared columns: {unexpected}")


def _validate_contract_units(table, contract):
    """Validate physical units declared by a serialization contract."""
    for name, unit in contract.get("units", {}).items():
        if name not in table.colnames:
            continue
        actual = getattr(table[name], "unit", None)
        if unit == "dimensionless" and actual is None:
            continue
        try:
            matches = actual is not None and u.Unit(actual) == u.Unit(unit)
        except (TypeError, ValueError, u.UnitsError) as exc:
            raise ValueError(
                f"sim_telarray column '{name}' has unit {actual}; expected {unit}"
            ) from exc
        if not matches:
            raise ValueError(f"sim_telarray column '{name}' has unit {actual}; expected {unit}")


def _validate_contract_matrix(table, contract):
    """Validate matrix-axis presence, uniqueness, and Cartesian completeness."""
    matrix_axes = contract.get("matrix_axes", [])
    present_axes = [axis for axis in matrix_axes if axis in table.colnames]
    if (
        contract.get("table_format") == "rpol_matrix"
        and len(matrix_axes) == 2
        and present_axes == [matrix_axes[0]]
    ):
        present_axes = []
    if present_axes and len(present_axes) != len(matrix_axes):
        raise ValueError("sim_telarray matrix axes must be present together")
    if present_axes:
        keys = list(zip(*(list(_raw_values(table[axis])) for axis in matrix_axes), strict=True))
        if len(keys) != len(set(keys)):
            raise ValueError("sim_telarray matrix contains duplicate axis combinations")
        expected = {
            tuple(values)
            for values in product(*[sorted(set(_raw_values(table[axis]))) for axis in matrix_axes])
        }
        if set(keys) != expected:
            raise ValueError("sim_telarray matrix must contain a complete Cartesian grid")


def _ordered_table(table, contract):
    """Return a selected, sorted table view without mutating the source."""
    columns = list(contract["columns"])
    columns.extend(name for name in contract.get("optional_columns", []) if name in table.colnames)
    result = table[columns].copy(copy_data=True)
    sort_keys = [
        name
        for name in contract.get("row_sort_keys", contract.get("sort_keys", []))
        if name in result.colnames
    ]
    if sort_keys:
        result.sort(sort_keys)
    return result


def _format_value(value, contract):
    """Format a scalar according to the contract."""
    value = getattr(value, "value", value)
    if isinstance(value, (float, np.floating)):
        return format(float(value), contract.get("float_format", ".12g"))
    return str(value)


def _column_values(table, name, contract):
    """Return formatted values for one declared output column."""
    return [_format_value(value, contract) for value in table[name]]


def _write_plain_table(table, output_path, contract):
    """Write selected, sorted columns as whitespace-separated rows."""
    ordered = _ordered_table(table, contract)
    selected = list(contract["columns"])
    selected.extend(
        name for name in contract.get("optional_columns", []) if name in ordered.colnames
    )
    columns = [_column_values(ordered, name, contract) for name in selected]
    with output_path.open("w", encoding="utf-8") as file:
        for row in zip(*columns, strict=True):
            file.write(" ".join(row) + "\n")


def _write_rpol_table(table, output_path, contract):
    """Write a tidy wavelength/angle table in sim_telarray RPOL format."""
    axes = contract.get("matrix_axes", [])
    if len(axes) != 2 or not all(axis in table.colnames for axis in axes):
        _write_plain_table(table, output_path, contract)
        return
    value_column = contract["value_column"]
    axis_0, axis_1 = axes
    values = {
        (
            getattr(row[axis_0], "value", row[axis_0]),
            getattr(row[axis_1], "value", row[axis_1]),
        ): getattr(row[value_column], "value", row[value_column])
        for row in table
    }
    axis_1_values = sorted(set(_raw_values(table[axis_1])))
    axis_0_values = sorted(set(_raw_values(table[axis_0])))
    with output_path.open("w", encoding="utf-8") as file:
        file.write("#@RPOL@[ANGLE=] 2\n")
        file.write(
            "ANGLE= " + " ".join(_format_value(value, contract) for value in axis_1_values) + "\n"
        )
        for axis_0_value in axis_0_values:
            row = [values[(axis_0_value, axis_1_value)] for axis_1_value in axis_1_values]
            fields = [_format_value(axis_0_value, contract)]
            fields.extend(_format_value(value, contract) for value in row)
            file.write(" ".join(fields) + "\n")


def _write_atmospheric_transmission(table, output_path, contract):
    """Write atmospheric transmission in sim_telarray matrix format."""
    axis_0, axis_1 = contract.get("matrix_axes", ["wavelength", "altitude"])
    value_column = contract["value_column"]
    axis_0_values = sorted(set(_raw_values(table[axis_0])))
    axis_1_values = sorted(set(_raw_values(table[axis_1])))
    values = {
        (
            getattr(row[axis_0], "value", row[axis_0]),
            getattr(row[axis_1], "value", row[axis_1]),
        ): getattr(row[value_column], "value", row[value_column])
        for row in table
    }
    observatory_level = table.meta.get("observatory_level")
    observatory_level = getattr(observatory_level, "value", observatory_level)
    with output_path.open("w", encoding="utf-8") as file:
        header = "# H1= " + " ".join(_format_value(value, contract) for value in axis_1_values)
        if observatory_level is not None:
            header = f"# H2= {_format_value(observatory_level, contract)}, {header[2:]}"
        file.write(header + "\n")
        for axis_0_value in axis_0_values:
            fields = [_format_value(axis_0_value, contract)]
            fields.extend(
                _format_value(values[(axis_0_value, axis_1_value)], contract)
                for axis_1_value in axis_1_values
            )
            file.write(" ".join(fields) + "\n")


def _raw_values(values):
    """Return plain scalar values from an Astropy column or iterable."""
    return [getattr(value, "value", value) for value in values]


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
