"""Tests for integration-output artifacts."""

from pathlib import Path

import pytest

from simtools.testing.output_validation.artifacts import OutputArtifact


def test_output_artifact_resolves_and_checks_existence(tmp_test_directory):
    """Resolve an output descriptor and accept an existing artifact."""
    output = Path(tmp_test_directory) / "nested" / "result.txt"
    output.parent.mkdir()
    output.touch()

    artifact = OutputArtifact.from_descriptor(
        {"output_path": tmp_test_directory},
        {"path_descriptor": "output_path", "output_sub_path": "nested", "file": output.name},
    )

    artifact.assert_exists()
    assert artifact.path == output


def test_output_artifact_reports_missing_configuration(tmp_test_directory):
    """Report a missing configured path descriptor."""
    with pytest.raises(KeyError, match="Path missing"):
        OutputArtifact.from_descriptor(
            {"output_path": tmp_test_directory},
            {"path_descriptor": "missing", "file": "result.txt"},
        )
