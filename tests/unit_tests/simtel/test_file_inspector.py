"""Tests for sim_telarray output inspection."""

import pytest

from simtools.simtel.file_inspector import (
    _format_simtel_report,
    inspect_sim_telarray_file,
)


def test_inspect_sim_telarray_file_rejects_suffix_inconsistent_with_type(tmp_path, mocker):
    file_path = tmp_path / "run.simtel.zst"
    file_path.write_bytes(b"simtel")
    mocker.patch("simtools.simtel.file_inspector.is_file_type", return_value=False)

    with pytest.raises(ValueError, match="unsupported suffix for sim_telarray inspection"):
        inspect_sim_telarray_file(file_path)


def test_inspect_sim_telarray_file_formats_metadata(tmp_path, mocker):
    file_path = tmp_path / "run.simtel.zst"
    file_path.write_bytes(b"simtel")
    mocker.patch(
        "simtools.simtel.file_inspector.simtel_io_metadata.read_sim_telarray_metadata",
        return_value=({"site": "North"}, {7: {"camera": "LST"}}),
    )

    report = inspect_sim_telarray_file(file_path)

    assert "global_metadata:" in report
    assert "- site: North" in report
    assert "telescope_count: 1" in report
    assert "telescope_ids: [7]" in report


def test_format_simtel_report_formats_truncated_metadata(tmp_path):
    report = _format_simtel_report(
        {
            "file_path": tmp_path / "run.simtel.zst",
            "file_type": "sim_telarray",
            "file_size_bytes": 6,
            "telescope_count": 3,
            "telescope_ids": [1, 2],
            "global_metadata": {"site": "North"},
            "global_metadata_truncated": True,
            "telescope_ids_truncated": True,
        }
    )

    assert "- ... metadata truncated ..." in report
    assert "telescope_ids_truncated: True" in report
