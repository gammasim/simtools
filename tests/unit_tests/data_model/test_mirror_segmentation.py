"""Tests for mirror-segmentation validation and serialization."""

from pathlib import Path

import pytest

from simtools.data_model.mirror_segmentation import (
    parse_segmentation_file,
    validate_segments,
    write_mirror_segmentation,
)

PARAMETER_NAME = "primary_mirror_segmentation"
SCHEMA_VERSION = "0.2.0"


def test_parse_and_write_ring_segments(tmp_test_directory):
    source = Path(tmp_test_directory) / "segments.dat"
    source.write_text("# comment\nRING 2 1 2 90 0 0.1\n", encoding="utf-8")
    records = parse_segmentation_file(source, PARAMETER_NAME, SCHEMA_VERSION)
    output = Path(tmp_test_directory) / "output.dat"

    write_mirror_segmentation(records, output, PARAMETER_NAME, SCHEMA_VERSION)

    assert output.read_text(encoding="utf-8").splitlines()[-1] == "RING 2 1.0 2.0 90.0 0.0 0.1"


def test_parse_and_write_shape_and_polygon_segments(tmp_test_directory):
    source = Path(tmp_test_directory) / "segments.dat"
    source.write_text(
        "HEX 1 1 2 3 4\nPOLYGON 1 5 0 0 1 0 0 1\n",
        encoding="utf-8",
    )
    records = parse_segmentation_file(source, PARAMETER_NAME, SCHEMA_VERSION)
    output = Path(tmp_test_directory) / "output.dat"

    write_mirror_segmentation(records, output, PARAMETER_NAME, SCHEMA_VERSION)

    assert output.read_text(encoding="utf-8").splitlines()[1:] == [
        "HEX 1 1.0 2.0 3.0 4.0",
        "POLYGON 1 5.0 0.0 0.0 1.0 0.0 0.0 1.0",
    ]


def test_validate_shape_and_polygon():
    validate_segments(
        [
            {"kind": "hex", "x_cm": 0, "y_cm": 0, "diameter_cm": 10},
            {
                "kind": "polygon",
                "vertices_cm": [
                    {"x_cm": 0, "y_cm": 0},
                    {"x_cm": 1, "y_cm": 0},
                    {"x_cm": 0, "y_cm": 1},
                ],
            },
        ],
        PARAMETER_NAME,
        SCHEMA_VERSION,
    )


@pytest.mark.parametrize(
    "record",
    [
        {"kind": "triangle", "x_cm": 0, "y_cm": 0, "diameter_cm": 10},
        {"kind": "hex", "x_cm": 0, "y_cm": 0, "diameter_cm": 10, "extra": 1},
        {"kind": "hex", "count": 2, "x_cm": 0, "y_cm": 0, "diameter_cm": 10},
    ],
)
def test_validate_rejects_schema_invalid_record(record):
    with pytest.raises(ValueError, match="Invalid mirror segmentation"):
        validate_segments([record], PARAMETER_NAME, SCHEMA_VERSION)


def test_validate_uses_selected_parameter_schema():
    validate_segments(
        [{"kind": "circular", "x_cm": 0, "y_cm": 0, "diameter_cm": 10}],
        parameter_name="secondary_mirror_segmentation",
        schema_version="0.2.0",
    )


def test_validate_rejects_invalid_ring_and_polygon():
    with pytest.raises(ValueError, match="greater than"):
        validate_segments(
            [{"kind": "ring", "count": 1, "r_min_cm": 2, "r_max_cm": 1, "dphi_deg": 1}],
            PARAMETER_NAME,
            SCHEMA_VERSION,
        )
    with pytest.raises(ValueError, match="non-zero area"):
        validate_segments(
            [
                {
                    "kind": "polygon",
                    "vertices_cm": [
                        {"x_cm": 0, "y_cm": 0},
                        {"x_cm": 1, "y_cm": 1},
                        {"x_cm": 2, "y_cm": 2},
                    ],
                }
            ],
            PARAMETER_NAME,
            SCHEMA_VERSION,
        )


def test_validate_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="finite"):
        validate_segments(
            [{"kind": "hex", "x_cm": float("nan"), "y_cm": 0, "diameter_cm": 10}],
            PARAMETER_NAME,
            SCHEMA_VERSION,
        )


def test_write_rejects_path_traversal(tmp_test_directory):
    with pytest.raises(ValueError, match="Unsafe"):
        write_mirror_segmentation(
            [{"kind": "hex", "x_cm": 0, "y_cm": 0, "diameter_cm": 1}],
            Path(tmp_test_directory) / ".." / "bad.dat",
            PARAMETER_NAME,
            SCHEMA_VERSION,
        )
