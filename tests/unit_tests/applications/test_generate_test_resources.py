"""Tests for the resource-generation application."""

from simtools.applications import generate_test_resources


def test_application_does_not_initialize_model_reader():
    """Resource orchestration must not require an external model source during startup."""
    assert "model" not in generate_test_resources.APPLICATION.needs
