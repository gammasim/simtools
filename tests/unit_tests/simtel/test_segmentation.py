"""Tests for sim_telarray mirror-segmentation serialization."""

from pathlib import Path

import pytest

from simtools.simtel.segmentation import write_mirror_segmentation

PARAMETER_NAME = "primary_mirror_segmentation"
SCHEMA_VERSION = "0.2.0"


def test_write_mirror_segmentation_serializes_all_supported_shapes(tmp_test_directory):
    output = Path(tmp_test_directory) / "segments.dat"
    records = [
        {
            "kind": "hex",
            "x": {"value": 1, "unit": "cm"},
            "y": {"value": 2, "unit": "cm"},
            "diameter": {"value": 3, "unit": "cm"},
            "rotation": {"value": 4, "unit": "deg"},
        },
        {
            "kind": "polygon",
            "vertices": [
                {"x": {"value": 0, "unit": "cm"}, "y": {"value": 0, "unit": "cm"}},
                {"x": {"value": 1, "unit": "cm"}, "y": {"value": 0, "unit": "cm"}},
                {"x": {"value": 0, "unit": "cm"}, "y": {"value": 1, "unit": "cm"}},
            ],
            "rotation": {"value": 5, "unit": "deg"},
        },
    ]

    result = write_mirror_segmentation(records, output, PARAMETER_NAME, SCHEMA_VERSION)

    assert result == "segments.dat"
    assert output.read_text(encoding="utf-8").splitlines() == [
        "HEX 1 1.0 2.0 3.0 4.0",
        "POLYGON 1 5.0 0.0 0.0 1.0 0.0 0.0 1.0",
    ]


def test_write_mirror_segmentation_rejects_path_traversal(tmp_test_directory):
    with pytest.raises(ValueError, match="Unsafe"):
        write_mirror_segmentation(
            [
                {
                    "kind": "hex",
                    "x": {"value": 0, "unit": "cm"},
                    "y": {"value": 0, "unit": "cm"},
                    "diameter": {"value": 1, "unit": "cm"},
                }
            ],
            Path(tmp_test_directory) / ".." / "segments.dat",
            PARAMETER_NAME,
            SCHEMA_VERSION,
        )
