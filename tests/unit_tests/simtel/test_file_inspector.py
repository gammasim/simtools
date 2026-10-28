"""Tests for sim_telarray output inspection."""

from pathlib import Path

import pytest

from simtools.simtel.file_inspector import (
    _format_simtel_report,
    _is_truncated,
    inspect_sim_telarray_file,
)


def _report(tmp_test_directory, **overrides):
    report = {
        "file_path": Path(tmp_test_directory) / "run.simtel.zst",
        "file_type": "sim_telarray",
        "file_size_bytes": 6,
        "telescope_count": 1,
        "telescope_ids": [7],
        "global_metadata": {"site": "North"},
        "global_metadata_truncated": False,
        "telescope_ids_truncated": False,
    }
    report.update(overrides)
    return report


def test_inspect_sim_telarray_file_rejects_unsupported_suffix(tmp_test_directory):
    file_path = Path(tmp_test_directory) / "run.txt"
    file_path.write_text("not a sim_telarray file", encoding="utf-8")

    with pytest.raises(ValueError, match=r"has unsupported suffix, expected one of"):
        inspect_sim_telarray_file(file_path)


def test_format_simtel_report_lists_metadata_and_telescopes(tmp_test_directory):
    report = _format_simtel_report(_report(tmp_test_directory))

    assert "Detected file type: sim_telarray" in report
    assert "global_metadata:" in report
    assert "- site: North" in report
    assert "telescope_count: 1" in report
    assert "telescope_ids: [7]" in report
    assert "truncated" not in report


def test_format_simtel_report_without_metadata(tmp_test_directory):
    report = _format_simtel_report(_report(tmp_test_directory, global_metadata={}))

    assert "global_metadata:\n- none" in report


def test_format_simtel_report_formats_truncated_metadata(tmp_test_directory):
    report = _format_simtel_report(
        _report(
            tmp_test_directory,
            telescope_count=3,
            telescope_ids=[1, 2],
            global_metadata_truncated=True,
            telescope_ids_truncated=True,
        )
    )

    assert "- ... metadata truncated ..." in report
    assert "telescope_ids_truncated: True" in report


@pytest.mark.parametrize(
    ("max_entries", "expected"),
    [(None, False), (0, False), (2, True), (3, False)],
)
def test_is_truncated(max_entries, expected):
    assert _is_truncated([1, 2, 3], max_entries) is expected
