"""Tests for the docs_produce_production_summary application."""

from simtools.applications import docs_produce_production_summary


def test_application_does_not_initialize_model_reader():
    """Production summaries read the supplied repository path directly."""
    assert docs_produce_production_summary.APPLICATION.initialize_model_reader is False
