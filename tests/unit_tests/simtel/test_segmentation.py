"""Tests for sim_telarray mirror-segmentation serialization."""

from pathlib import Path

import pytest

from simtools.simtel.segmentation import parse_segmentation_file, write_mirror_segmentation

PARAMETER_NAME = "primary_mirror_segmentation"
SCHEMA_VERSION = "0.2.0"


def test_parse_segmentation_file_ignores_comments_and_commas(tmp_test_directory):
    source = Path(tmp_test_directory) / "segments.dat"
    source.write_text(
        "# header\nRING, 2, 1, 2, 90, 0, 0.1 # trailing comment\n",
        encoding="utf-8",
    )

    records = parse_segmentation_file(source, PARAMETER_NAME, SCHEMA_VERSION)

    assert records == [
        {
            "kind": "ring",
            "count": 2,
            "r_min_cm": 1.0,
            "r_max_cm": 2.0,
            "dphi_deg": 90.0,
            "phi0_deg": 0.0,
            "gap_cm": 0.1,
        }
    ]


def test_write_mirror_segmentation_serializes_all_supported_shapes(tmp_test_directory):
    output = Path(tmp_test_directory) / "segments.dat"
    records = [
        {"kind": "hex", "x_cm": 1, "y_cm": 2, "diameter_cm": 3, "rotation_deg": 4},
        {
            "kind": "polygon",
            "vertices_cm": [
                {"x_cm": 0, "y_cm": 0},
                {"x_cm": 1, "y_cm": 0},
                {"x_cm": 0, "y_cm": 1},
            ],
            "rotation_deg": 5,
        },
    ]

    result = write_mirror_segmentation(records, output, PARAMETER_NAME, SCHEMA_VERSION)

    assert result == "segments.dat"
    assert output.read_text(encoding="utf-8").splitlines() == [
        "HEX 1 1 2 3 4",
        "POLYGON 1 5 0 0 1 0 0 1",
    ]


def test_parse_segmentation_file_rejects_unknown_kind(tmp_test_directory):
    source = Path(tmp_test_directory) / "segments.dat"
    source.write_text("OCTAGON 1 0 0 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown mirror segmentation kind"):
        parse_segmentation_file(source, PARAMETER_NAME, SCHEMA_VERSION)


def test_write_mirror_segmentation_rejects_path_traversal(tmp_test_directory):
    with pytest.raises(ValueError, match="Unsafe"):
        write_mirror_segmentation(
            [{"kind": "hex", "x_cm": 0, "y_cm": 0, "diameter_cm": 1}],
            Path(tmp_test_directory) / ".." / "segments.dat",
            PARAMETER_NAME,
            SCHEMA_VERSION,
        )
