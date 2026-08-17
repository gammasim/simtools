"""Tests for the public output-validation facade."""

from simtools.testing import validate_output
from simtools.testing.output_validation import runner


def test_validate_output_exposes_runner_entry_point():
    """Expose the runner's application validation function publicly."""
    assert validate_output.validate_application_output is runner.validate_application_output
