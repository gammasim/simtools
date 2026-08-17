"""Tests for sim_telarray output validators."""

from pathlib import Path

import pytest

from simtools.testing.output_validation import simtel


def test_compare_config_files_ignores_simtools_assignments(tmp_test_directory, mocker):
    """Ignore simtools-specific assignments when comparing configurations."""
    reference_file = Path(tmp_test_directory) / "reference.cfg"
    output_file = Path(tmp_test_directory) / "output.cfg"
    reference_file.write_text("simtools_version = 1\nparameter = 1\n", encoding="utf-8")
    output_file.write_text("simtools_version = 2\nparameter = 1\n", encoding="utf-8")
    mocker.patch.object(simtel, "_assignment_metadata_keys", return_value=set())

    assert simtel.compare_config_files(reference_file, output_file)


def test_simtel_output_validator_builds_expected_output(mocker):
    """Translate event expectations to the sim_telarray assertion format."""
    check = mocker.patch.object(
        simtel.assertions, "check_output_from_sim_telarray", return_value=True
    )

    simtel.validate_output(
        Path("output.simtel.zst"),
        {"event_type": "shower", "event": {"pe_sum": {"range": [20, 1000]}}},
    )

    check.assert_called_once_with(
        Path("output.simtel.zst"),
        {
            "expected_sim_telarray_output": {
                "event_type": "shower",
                "pe_sum": [20, 1000],
            }
        },
    )


def test_simtel_output_validator_includes_metadata_and_reports_failure(mocker):
    """Include metadata expectations and report failed sim_telarray checks."""
    check = mocker.patch.object(
        simtel.assertions, "check_output_from_sim_telarray", return_value=False
    )
    rule = {
        "event": {"pe_sum": {"range": [20, 1000]}},
        "metadata": {"run_number": 1},
    }

    with pytest.raises(AssertionError, match="failed sim_telarray validation"):
        simtel.validate_output(Path("output.simtel.zst"), rule)

    check.assert_called_once_with(
        Path("output.simtel.zst"),
        {
            "expected_sim_telarray_output": {
                "event_type": "shower",
                "pe_sum": [20, 1000],
            },
            "expected_sim_telarray_metadata": {"run_number": 1},
        },
    )


def test_assignment_metadata_keys_returns_assign_parameters(mocker):
    """Select sim_telarray metadata parameters represented as assignments."""
    mocker.patch.object(
        simtel.simtel_validate_metadata,
        "get_meta_parameter_registry",
        return_value={
            "meta_parameters": {
                "assigned": {"mode": "assign"},
                "checked": {"mode": "validate"},
            }
        },
    )

    assert simtel._assignment_metadata_keys() == {"assigned"}


def test_split_lines_handles_comments_metaparameters_metadata_and_controls(mocker):
    """Classify sim_telarray configuration lines."""
    mocker.patch.object(simtel, "_assignment_metadata_keys", return_value={"meta_key"})

    parameters, controls = simtel._split_lines(
        [
            "",
            "% comment",
            "metaparam global set ignored = 1",
            "meta_key = 2",
            "parameter = 3",
            "control directive",
        ],
        "config.cfg",
    )

    assert parameters == {"parameter": "3"}
    assert controls == ["control directive"]


def test_simtel_config_validator_detects_assignment_difference(tmp_test_directory, mocker):
    """Detect differences in sim_telarray assignments."""
    reference_file = Path(tmp_test_directory) / "reference.cfg"
    output_file = Path(tmp_test_directory) / "output.cfg"
    reference_file.write_text("parameter = 1\n", encoding="utf-8")
    output_file.write_text("parameter = 2\n", encoding="utf-8")
    mocker.patch.object(simtel, "_assignment_metadata_keys", return_value=set())

    assert not simtel.compare_config_files(reference_file, output_file)
