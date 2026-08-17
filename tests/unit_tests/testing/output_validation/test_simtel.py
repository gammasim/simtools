"""Tests for sim_telarray output validators."""

from pathlib import Path

from simtools.testing.output_validation import simtel


def test_compare_config_files_compares_assignments(tmp_test_directory, mocker):
    """Compare sim_telarray assignments while ignoring no metadata keys."""
    reference_file = Path(tmp_test_directory) / "reference.cfg"
    output_file = Path(tmp_test_directory) / "output.cfg"
    reference_file.write_text("parameter = 1\n", encoding="utf-8")
    output_file.write_text("parameter = 1\n", encoding="utf-8")
    mocker.patch.object(simtel, "_assignment_metadata_keys", return_value=set())

    assert simtel.compare_config_files(reference_file, output_file)
