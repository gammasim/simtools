"""Tests for the resource-generation application."""

from simtools.applications import resources_test_generate


def test_application_does_not_initialize_model_reader():
    """Resource orchestration must not require MongoDB during startup."""
    assert resources_test_generate.APPLICATION.initialize_model_reader is False
