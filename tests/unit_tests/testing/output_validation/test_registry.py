"""Tests for output-validator registration and dispatch."""

from pathlib import Path

import pytest

from simtools.testing.output_validation import registry
from simtools.testing.output_validation.artifacts import OutputArtifact


def test_registry_rejects_unknown_validator():
    """Reject a validation rule with an unknown type."""
    with pytest.raises(ValueError, match="Unknown output validator"):
        registry.run_validator(OutputArtifact(Path("output"), {}), {"type": "missing"}, {})
