"""Tests for output-validation orchestration."""

from simtools.testing.output_validation import runner


def test_versions_match_requires_an_intersection():
    """Match scalar and sequence model-version filters."""
    assert runner.versions_match(None, "7.0.0")
    assert runner.versions_match("7.0.0", ["6.0.0", "7.0.0"])
    assert not runner.versions_match("7.0.0", "6.0.0")
