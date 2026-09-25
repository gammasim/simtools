"""Validation and serialization of mirror-segmentation model parameters."""

import math
from functools import lru_cache

import astropy.units as u
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
        if "r_min" in required_fields:
            _validate_ring(record)
        elif "vertices" in required_fields:
            _validate_polygon(record)
        else:
            _validate_shape(record)
    return records


def quantity_value(record, field, unit, default=None):
    """Return one segmentation quantity converted to the requested unit.

    Parameters
    ----------
    record : dict
        Segmentation record containing an explicit ``value``/``unit`` object.
    field : str
        Quantity field name.
    unit : str
        Unit to which the value is converted.
    default : float, optional
        Value returned when the optional field is absent.

    Returns
    -------
    float
        Numeric value in ``unit``.
    """
    quantity = record.get(field)
    if quantity is None:
        return default
    try:
        return (quantity["value"] * u.Unit(quantity["unit"])).to(unit).value
    except (KeyError, TypeError, ValueError, u.UnitConversionError) as exc:
        raise ValueError(f"Invalid unit-bearing segmentation field '{field}'") from exc


def make_quantity(value, unit):
    """Return a JSON-compatible explicit quantity object.

    Parameters
    ----------
    value : float
        Numeric quantity value.
    unit : str
        Unit associated with ``value``.

    Returns
    -------
    dict
        Quantity represented by ``value`` and ``unit`` keys.
    """
    return {"value": value, "unit": unit}


def _validate_ring(record):
    r_min = quantity_value(record, "r_min", "cm")
    r_max = quantity_value(record, "r_max", "cm")
    if r_min < 0 or r_max <= 0 or r_max <= r_min:
        raise ValueError("Ring r_max must be greater than r_min")
    if quantity_value(record, "dphi", "deg") <= 0:
        raise ValueError("Ring dphi must be positive")
    if quantity_value(record, "gap", "cm", 0) < 0:
        raise ValueError("Ring gap must not be negative")


def _validate_shape(record):
    """Validate the positive size of a shaped mirror facet."""
    if quantity_value(record, "diameter", "cm") <= 0:
        raise ValueError("Shape diameter must be positive")


def _validate_polygon(record):
    vertices = record.get("vertices")
    vertices = [
        {"x": quantity_value(vertex, "x", "cm"), "y": quantity_value(vertex, "y", "cm")}
        for vertex in vertices
    ]
    area = sum(
        first["x"] * second["y"] - second["x"] * first["y"]
        for first, second in zip(vertices, vertices[1:] + vertices[:1], strict=False)
    )
    if math.isclose(area, 0, abs_tol=1e-12):
        raise ValueError("Polygon vertices must enclose a non-zero area")
