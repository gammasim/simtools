"""Validation and serialization of mirror-segmentation model parameters."""

import math
from functools import lru_cache
from pathlib import Path

from simtools.data_model import schema


@lru_cache
def _shape_kinds():
    """Return shape kinds declared by the mirror-segmentation schemas."""
    kinds = set()
    for parameter in ("primary_mirror_segmentation", "secondary_mirror_segmentation"):
        parameter_schema = schema.get_model_parameter_schema(parameter, "0.2.0")
        json_schema = next(
            item["json_schema"] for item in parameter_schema["data"] if item["type"] == "dict"
        )
        for item in json_schema["items"]["oneOf"]:
            kinds.update(item.get("properties", {}).get("kind", {}).get("enum", []))
    return frozenset(kinds - {"ring", "polygon"})


def validate_segments(records):
    """Validate mirror-segmentation records and return them unchanged.

    Parameters
    ----------
    records : list of dict
        Ring, shape, or polygon records.

    Returns
    -------
    list of dict
        The validated records.

    Raises
    ------
    ValueError
        If a record is malformed or contains non-finite geometry.
    """
    if not isinstance(records, list) or not records:
        raise ValueError("Mirror segmentation must contain at least one record")
    for record in records:
        if not isinstance(record, dict) or set(record) - _allowed_keys(record):
            raise ValueError(f"Invalid mirror segmentation record: {record!r}")
        kind = record.get("kind")
        if kind == "ring":
            _validate_ring(record)
        elif kind in _shape_kinds():
            _validate_shape(record)
        elif kind == "polygon":
            _validate_polygon(record)
        else:
            raise ValueError(f"Unknown mirror segmentation kind: {kind!r}")
    return records


def _allowed_keys(record):
    common = {"kind", "count", "rotation_deg"}
    if record.get("kind") == "ring":
        return common | {"r_min_cm", "r_max_cm", "dphi_deg", "phi0_deg", "gap_cm"}
    if record.get("kind") == "polygon":
        return common | {"vertices_cm"}
    return common | {"x_cm", "y_cm", "diameter_cm"}


def _number(record, key, minimum=None, positive=False):
    value = record.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"Mirror segmentation field '{key}' must be finite numeric")
    if positive and value <= 0:
        raise ValueError(f"Mirror segmentation field '{key}' must be positive")
    if minimum is not None and value < minimum:
        raise ValueError(f"Mirror segmentation field '{key}' must be >= {minimum}")
    return value


def _optional_number(record, key, minimum=None):
    if key in record:
        _number(record, key, minimum=minimum)


def _validate_ring(record):
    count = record.get("count")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("Ring count must be a positive integer")
    r_min = _number(record, "r_min_cm", minimum=0)
    r_max = _number(record, "r_max_cm", minimum=0)
    if r_max <= r_min:
        raise ValueError("Ring r_max_cm must be greater than r_min_cm")
    _number(record, "dphi_deg", positive=True)
    _optional_number(record, "phi0_deg")
    _optional_number(record, "gap_cm", minimum=0)


def _validate_shape(record):
    if record.get("count", 1) != 1:
        raise ValueError("Individual mirror shapes must have count=1")
    _number(record, "x_cm")
    _number(record, "y_cm")
    _number(record, "diameter_cm", positive=True)
    _optional_number(record, "rotation_deg")


def _validate_polygon(record):
    if record.get("count", 1) != 1:
        raise ValueError("Individual polygons must have count=1")
    vertices = record.get("vertices_cm")
    if not isinstance(vertices, list) or len(vertices) < 3:
        raise ValueError("A polygon must contain at least three vertices")
    for vertex in vertices:
        if not isinstance(vertex, dict) or set(vertex) != {"x_cm", "y_cm"}:
            raise ValueError("Polygon vertices must contain only x_cm and y_cm")
        _number(vertex, "x_cm")
        _number(vertex, "y_cm")
    _optional_number(record, "rotation_deg")
    area = sum(
        first["x_cm"] * second["y_cm"] - second["x_cm"] * first["y_cm"]
        for first, second in zip(vertices, vertices[1:] + vertices[:1], strict=False)
    )
    if math.isclose(area, 0):
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


def _parse_segmentation_line(line):
    fields = line.replace(",", " ").split()
    kind = fields.pop(0).lower()
    count = int(fields.pop(0))
    if kind == "ring":
        return _parse_ring(fields, line, kind, count)
    if kind in _shape_kinds():
        return _parse_shape(fields, line, kind, count)
    if kind == "polygon":
        return _parse_polygon(fields, line, kind, count)
    raise ValueError(f"Unknown mirror segmentation kind: {kind}")


def parse_segmentation_file(path):
    """Parse a legacy sim_telarray mirror-segmentation file into records."""
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            records.append(_parse_segmentation_line(line))
    return validate_segments(records)


def write_mirror_segmentation(records, output_path):
    """Write validated records in sim_telarray segmentation syntax."""
    validate_segments(records)
    output_path = Path(output_path)
    if ".." in output_path.parts:
        raise ValueError(f"Unsafe mirror segmentation output path: {output_path}")
    lines = ["# Generated by simtools from validated mirror segmentation records"]
    for record in records:
        kind = record["kind"]
        if kind == "ring":
            lines.append(
                f"RING {record['count']} {record['r_min_cm']} {record['r_max_cm']} "
                f"{record['dphi_deg']} {record.get('phi0_deg', 0)} {record.get('gap_cm', 0)}"
            )
        elif kind in _shape_kinds():
            lines.append(
                f"{kind.upper()} 1 {record['x_cm']} {record['y_cm']} "
                f"{record['diameter_cm']} {record.get('rotation_deg', 0)}"
            )
        else:
            vertices = " ".join(
                f"{vertex['x_cm']} {vertex['y_cm']}" for vertex in record["vertices_cm"]
            )
            lines.append(f"POLYGON 1 {record.get('rotation_deg', 0)} {vertices}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path.name
