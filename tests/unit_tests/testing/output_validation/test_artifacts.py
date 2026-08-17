"""Tests for integration-output artifacts."""

from pathlib import Path

import pytest

from simtools.testing.output_validation.artifacts import OutputArtifact


def test_output_artifact_uses_empty_output_subpath_by_default(tmp_test_directory):
    """Resolve an artifact when no output subpath is configured."""
    output = Path(tmp_test_directory) / "nested" / "result.txt"
    output.parent.mkdir()
    output.touch()

    artifact = OutputArtifact.from_descriptor(
        {"output_path": tmp_test_directory},
        {"path_descriptor": "output_path", "file": "nested/result.txt"},
    )

    artifact.assert_exists()
    assert artifact.path == output


def test_output_artifact_reports_configuration_and_missing_file(tmp_test_directory):
    """Report missing path configuration and missing artifacts."""
    with pytest.raises(KeyError, match="Path missing"):
        OutputArtifact.from_descriptor(
            {"output_path": tmp_test_directory},
            {"path_descriptor": "missing", "file": "result.txt"},
        )
    artifact = OutputArtifact(Path(tmp_test_directory) / "missing.txt", {})
    with pytest.raises(AssertionError, match="does not exist"):
        artifact.assert_exists()
