"""Validation and serialization of mirror-segmentation model parameters."""

import math
from functools import lru_cache
from pathlib import Path

from jsonschema.exceptions import ValidationError

from simtools.data_model import schema


@lru_cache
def _segmentation_json_schema(parameter_name, schema_version):
    """Return the JSON schema for a mirror-segmentation parameter."""
    parameter_schema = schema.get_model_parameter_schema(parameter_name, schema_version)
    return next(item["json_schema"] for item in parameter_schema["data"] if item["type"] == "dict")


@lru_cache
def _kind_required_fields(parameter_name, schema_version):
    """Map schema-declared segmentation kinds to their required fields."""
    definitions = {}
    for item in _segmentation_json_schema(parameter_name, schema_version)["items"]["oneOf"]:
        kinds = item["properties"]["kind"].get("enum", [])
        for kind in kinds:
            definitions[kind] = frozenset(item["required"])
    return definitions


def validate_segments(
    records,
    parameter_name,
    schema_version,
):
    """Validate mirror-segmentation records and return them unchanged.

    Parameters
    ----------
    records : list of dict
        Ring, shape, or polygon records.
    parameter_name : str
        Mirror-segmentation model parameter whose schema validates the records.
    schema_version : str
        Version of the selected parameter schema.

    Returns
    -------
    list of dict
        The validated records.

    Raises
    ------
    ValueError
        If a record is malformed or contains non-finite geometry.
    """
    try:
        schema.validate_dict_using_schema(
            records,
            json_schema=_segmentation_json_schema(parameter_name, schema_version),
        )
    except ValidationError as exc:
        raise ValueError(f"Invalid mirror segmentation: {exc.message}") from exc
    for record in records:
        _validate_finite_values(record)
        required_fields = _kind_required_fields(parameter_name, schema_version)[record["kind"]]
        if "r_min_cm" in required_fields:
            _validate_ring(record)
        elif "vertices_cm" in required_fields:
            _validate_polygon(record)
    return records


def _validate_finite_values(value):
    """Reject non-finite numbers, which JSON Schema does not constrain."""
    if isinstance(value, dict):
        for item in value.values():
            _validate_finite_values(item)
    elif isinstance(value, list):
        for item in value:
            _validate_finite_values(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Mirror segmentation values must be finite")


def _validate_ring(record):
    if record["r_max_cm"] <= record["r_min_cm"]:
        raise ValueError("Ring r_max_cm must be greater than r_min_cm")


def _validate_polygon(record):
    vertices = record.get("vertices_cm")
    area = sum(
        first["x_cm"] * second["y_cm"] - second["x_cm"] * first["y_cm"]
        for first, second in zip(vertices, vertices[1:] + vertices[:1], strict=False)
    )
    if math.isclose(area, 0, abs_tol=1e-12):
        raise ValueError("Polygon vertices must enclose a non-zero area")


def _parse_ring(fields, line, kind, count):
    if len(fields) not in (3, 4, 5):
        raise ValueError(f"Invalid ring segmentation line: {line}")
    return {
        "kind": kind,
        "count": count,
        "r_min_cm": float(fields[0]),
        "r_max_cm": float(fields[1]),
        "dphi_deg": float(fields[2]),
        "phi0_deg": float(fields[3]) if len(fields) > 3 else 0.0,
        "gap_cm": float(fields[4]) if len(fields) > 4 else 0.0,
    }


def _parse_shape(fields, line, kind, count):
    if len(fields) not in (3, 4):
        raise ValueError(f"Invalid shape segmentation line: {line}")
    return {
        "kind": kind,
        "count": count,
        "x_cm": float(fields[0]),
        "y_cm": float(fields[1]),
        "diameter_cm": float(fields[2]),
        "rotation_deg": float(fields[3]) if len(fields) == 4 else 0.0,
    }


def _parse_polygon(fields, line, kind, count):
    if len(fields) < 7 or len(fields[1:]) % 2:
        raise ValueError(f"Invalid polygon segmentation line: {line}")
    return {
        "kind": kind,
        "count": count,
        "rotation_deg": float(fields[0]),
        "vertices_cm": [
            {"x_cm": float(x), "y_cm": float(y)}
            for x, y in zip(fields[1::2], fields[2::2], strict=False)
        ],
    }


def _parse_segmentation_line(line, parameter_name, schema_version):
    fields = line.replace(",", " ").split()
    kind = fields.pop(0).lower()
    count = int(fields.pop(0))
    required_fields = _kind_required_fields(parameter_name, schema_version).get(kind)
    if required_fields is None:
        raise ValueError(f"Unknown mirror segmentation kind: {kind}")
    if "r_min_cm" in required_fields:
        return _parse_ring(fields, line, kind, count)
    if "vertices_cm" in required_fields:
        return _parse_polygon(fields, line, kind, count)
    return _parse_shape(fields, line, kind, count)


def parse_segmentation_file(
    path,
    parameter_name,
    schema_version,
):
    """Parse a sim_telarray mirror-segmentation file into validated records.

    Parameters
    ----------
    path : str or Path
        Input segmentation file.
    parameter_name : str
        Mirror-segmentation model parameter whose schema validates the records.
    schema_version : str
        Version of the selected parameter schema.
    """
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            records.append(_parse_segmentation_line(line, parameter_name, schema_version))
    return validate_segments(records, parameter_name, schema_version)


def write_mirror_segmentation(
    records,
    output_path,
    parameter_name,
    schema_version,
):
    """Write validated records in sim_telarray segmentation syntax.

    Parameters
    ----------
    records : list of dict
        Mirror-segmentation records.
    output_path : str or Path
        Output segmentation file.
    parameter_name : str
        Mirror-segmentation model parameter whose schema validates the records.
    schema_version : str
        Version of the selected parameter schema.

    Returns
    -------
    str
        Name of the output file.
    """
    validate_segments(records, parameter_name, schema_version)
    output_path = Path(output_path)
    if ".." in output_path.parts:
        raise ValueError(f"Unsafe mirror segmentation output path: {output_path}")
    lines = ["# Generated by simtools from validated mirror segmentation records"]
    for record in records:
        required_fields = _kind_required_fields(parameter_name, schema_version)[record["kind"]]
        if "r_min_cm" in required_fields:
            lines.append(
                f"RING {record['count']} {record['r_min_cm']} {record['r_max_cm']} "
                f"{record['dphi_deg']} {record.get('phi0_deg', 0)} {record.get('gap_cm', 0)}"
            )
        elif "vertices_cm" not in required_fields:
            lines.append(
                f"{record['kind'].upper()} 1 {record['x_cm']} {record['y_cm']} "
                f"{record['diameter_cm']} {record.get('rotation_deg', 0)}"
            )
        else:
            vertices = " ".join(
                f"{vertex['x_cm']} {vertex['y_cm']}" for vertex in record["vertices_cm"]
            )
            lines.append(f"POLYGON 1 {record.get('rotation_deg', 0)} {vertices}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path.name
