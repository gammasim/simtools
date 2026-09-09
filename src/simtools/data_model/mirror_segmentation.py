"""Validation and serialization of mirror-segmentation model parameters."""

import math
from functools import lru_cache

from jsonschema.exceptions import ValidationError

from simtools.data_model import schema
from simtools.data_model.json_validation import validate_finite_json_values


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
        validate_finite_json_values(record, path="$[record]")
        required_fields = _kind_required_fields(parameter_name, schema_version)[record["kind"]]
        if "r_min_cm" in required_fields:
            _validate_ring(record)
        elif "vertices_cm" in required_fields:
            _validate_polygon(record)
    return records


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
