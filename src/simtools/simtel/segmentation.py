"""sim_telarray segmentation serializer and historical-file adapter."""

from pathlib import Path

from simtools.data_model.mirror_segmentation import (
    _kind_required_fields,
    make_quantity,
    quantity_value,
    validate_segments,
)


def _parse_ring(fields, line, kind, count):
    if len(fields) not in (3, 4, 5):
        raise ValueError(f"Invalid ring segmentation line: {line}")
    return {
        "kind": kind,
        "count": count,
        "r_min": make_quantity(float(fields[0]), "cm"),
        "r_max": make_quantity(float(fields[1]), "cm"),
        "dphi": make_quantity(float(fields[2]), "deg"),
        "phi0": make_quantity(float(fields[3]) if len(fields) > 3 else 0.0, "deg"),
        "gap": make_quantity(float(fields[4]) if len(fields) > 4 else 0.0, "cm"),
    }


def _parse_shape(fields, line, kind, count):
    if len(fields) not in (3, 4):
        raise ValueError(f"Invalid shape segmentation line: {line}")
    return {
        "kind": kind,
        "count": count,
        "x": make_quantity(float(fields[0]), "cm"),
        "y": make_quantity(float(fields[1]), "cm"),
        "diameter": make_quantity(float(fields[2]), "cm"),
        "rotation": make_quantity(float(fields[3]) if len(fields) == 4 else 0.0, "deg"),
    }


def _parse_polygon(fields, line, kind, count):
    if len(fields) < 7 or len(fields[1:]) % 2:
        raise ValueError(f"Invalid polygon segmentation line: {line}")
    return {
        "kind": kind,
        "count": count,
        "rotation": make_quantity(float(fields[0]), "deg"),
        "vertices": [
            {"x": make_quantity(float(x), "cm"), "y": make_quantity(float(y), "cm")}
            for x, y in zip(fields[1::2], fields[2::2], strict=False)
        ],
    }


def _parse_line(line, parameter_name, schema_version):
    fields = line.replace(",", " ").split()
    kind = fields.pop(0).lower()
    count = int(fields.pop(0))
    required = _kind_required_fields(parameter_name, schema_version).get(kind)
    if required is None:
        raise ValueError(f"Unknown mirror segmentation kind: {kind}")
    if "r_min" in required:
        return _parse_ring(fields, line, kind, count)
    if "vertices" in required:
        return _parse_polygon(fields, line, kind, count)
    return _parse_shape(fields, line, kind, count)


def parse_segmentation_file(path, parameter_name, schema_version):
    """Parse a sim_telarray segmentation file for migration or diagnostics."""
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            records.append(_parse_line(line, parameter_name, schema_version))
    return validate_segments(records, parameter_name, schema_version)


def write_mirror_segmentation(records, output_path, parameter_name, schema_version):
    """Serialize validated segmentation records in sim_telarray syntax."""
    validate_segments(records, parameter_name, schema_version)
    output_path = Path(output_path)
    if ".." in output_path.parts:
        raise ValueError(f"Unsafe mirror segmentation output path: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for record in records:
        required = _kind_required_fields(parameter_name, schema_version)[record["kind"]]
        if "r_min" in required:
            lines.append(
                f"RING {record['count']} {quantity_value(record, 'r_min', 'cm')} "
                f"{quantity_value(record, 'r_max', 'cm')} "
                f"{quantity_value(record, 'dphi', 'deg')} "
                f"{quantity_value(record, 'phi0', 'deg', 0)} "
                f"{quantity_value(record, 'gap', 'cm', 0)}"
            )
        elif "vertices" not in required:
            lines.append(
                f"{record['kind'].upper()} 1 {quantity_value(record, 'x', 'cm')} "
                f"{quantity_value(record, 'y', 'cm')} {quantity_value(record, 'diameter', 'cm')} "
                f"{quantity_value(record, 'rotation', 'deg', 0)}"
            )
        else:
            vertices = " ".join(
                f"{quantity_value(vertex, 'x', 'cm')} {quantity_value(vertex, 'y', 'cm')}"
                for vertex in record["vertices"]
            )
            lines.append(f"POLYGON 1 {quantity_value(record, 'rotation', 'deg', 0)} {vertices}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path.name
