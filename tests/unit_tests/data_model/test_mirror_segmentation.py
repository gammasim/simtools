"""Tests for mirror-segmentation validation and serialization."""

from pathlib import Path

import pytest

from simtools.data_model.mirror_segmentation import validate_segments
from simtools.simtel.segmentation import parse_segmentation_file, write_mirror_segmentation

PARAMETER_NAME = "primary_mirror_segmentation"
SCHEMA_VERSION = "0.2.0"


def _q(value, unit="cm"):
    """Return an explicit JSON quantity."""
    return {"value": value, "unit": unit}


def _shape():
    """Return a valid shape record."""
    return {
        "kind": "hex",
        "count": 1,
        "x": _q(0),
        "y": _q(0),
        "diameter": _q(10),
        "rotation": _q(0, "deg"),
    }


def _polygon():
    """Return a valid polygon record."""
    return {
        "kind": "polygon",
        "count": 1,
        "rotation": _q(0, "deg"),
        "vertices": [
            {"x": _q(0), "y": _q(0)},
            {"x": _q(1), "y": _q(0)},
            {"x": _q(0), "y": _q(1)},
        ],
    }


def test_parse_and_write_ring_segments(tmp_test_directory):
    source = Path(tmp_test_directory) / "segments.dat"
    source.write_text("# comment\nRING 2 1 2 90 0 0.1\n", encoding="utf-8")
    records = parse_segmentation_file(source, PARAMETER_NAME, SCHEMA_VERSION)
    output = Path(tmp_test_directory) / "output.dat"

    write_mirror_segmentation(records, output, PARAMETER_NAME, SCHEMA_VERSION)

    assert output.read_text(encoding="utf-8").splitlines()[-1] == "RING 2 1.0 2.0 90.0 0.0 0.1"
    assert records[0]["r_min"] == _q(1.0)


def test_parse_and_write_shape_and_polygon_segments(tmp_test_directory):
    source = Path(tmp_test_directory) / "segments.dat"
    source.write_text(
        "HEX 1 1 2 3 4\nPOLYGON 1 5 0 0 1 0 0 1\n",
        encoding="utf-8",
    )
    records = parse_segmentation_file(source, PARAMETER_NAME, SCHEMA_VERSION)
    output = Path(tmp_test_directory) / "output.dat"

    write_mirror_segmentation(records, output, PARAMETER_NAME, SCHEMA_VERSION)

    assert output.read_text(encoding="utf-8").splitlines() == [
        "HEX 1 1.0 2.0 3.0 4.0",
        "POLYGON 1 5.0 0.0 0.0 1.0 0.0 0.0 1.0",
    ]


def test_validate_shape_and_polygon():
    validate_segments([_shape(), _polygon()], PARAMETER_NAME, SCHEMA_VERSION)


@pytest.mark.parametrize(
    "record",
    [
        {"kind": "triangle", "x": _q(0), "y": _q(0), "diameter": _q(10)},
        {**_shape(), "extra": 1},
        {**_shape(), "count": 2},
        {
            "kind": "hex",
            "count": 1,
            "x": {"value": 0, "unit": "deg"},
            "y": _q(0),
            "diameter": _q(10),
        },
    ],
)
def test_validate_rejects_schema_invalid_record(record):
    with pytest.raises(ValueError, match="Invalid mirror segmentation"):
        validate_segments([record], PARAMETER_NAME, SCHEMA_VERSION)


def test_validate_uses_selected_parameter_schema():
    validate_segments([_shape()], "secondary_mirror_segmentation", SCHEMA_VERSION)


def test_validate_accepts_compatible_units():
    record = _shape()
    record["diameter"] = _q(0.1, "m")
    validate_segments([record], PARAMETER_NAME, SCHEMA_VERSION)


def test_write_converts_compatible_units_to_simtel_centimeters(tmp_test_directory):
    record = _shape()
    record["diameter"] = _q(0.1, "m")
    output = Path(tmp_test_directory) / "segments.dat"

    write_mirror_segmentation([record], output, PARAMETER_NAME, SCHEMA_VERSION)

    assert output.read_text(encoding="utf-8").splitlines() == ["HEX 1 0.0 0.0 10.0 0.0"]


def test_validate_rejects_invalid_ring_and_polygon():
    with pytest.raises(ValueError, match="greater than"):
        validate_segments(
            [{"kind": "ring", "count": 1, "r_min": _q(2), "r_max": _q(1), "dphi": _q(1, "deg")}],
            PARAMETER_NAME,
            SCHEMA_VERSION,
        )
    polygon = _polygon()
    polygon["vertices"][2] = {"x": _q(2), "y": _q(0)}
    with pytest.raises(ValueError, match="non-zero area"):
        validate_segments([polygon], PARAMETER_NAME, SCHEMA_VERSION)


def test_validate_rejects_nonpositive_shape_diameter():
    record = _shape()
    record["diameter"] = _q(0)
    with pytest.raises(ValueError, match="diameter must be positive"):
        validate_segments([record], PARAMETER_NAME, SCHEMA_VERSION)


def test_validate_rejects_polygon_with_area_below_absolute_tolerance():
    polygon = _polygon()
    polygon["vertices"][2] = {"x": _q(0), "y": _q(1e-13)}
    with pytest.raises(ValueError, match="non-zero area"):
        validate_segments([polygon], PARAMETER_NAME, SCHEMA_VERSION)


def test_validate_rejects_nonfinite_values():
    record = _shape()
    record["x"]["value"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_segments([record], PARAMETER_NAME, SCHEMA_VERSION)


def test_write_rejects_path_traversal(tmp_test_directory):
    with pytest.raises(ValueError, match="Unsafe"):
        write_mirror_segmentation(
            [_shape()], Path(tmp_test_directory) / ".." / "bad.dat", PARAMETER_NAME, SCHEMA_VERSION
        )
