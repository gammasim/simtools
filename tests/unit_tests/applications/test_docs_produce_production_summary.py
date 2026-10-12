"""Tests for the docs_produce_production_summary application."""

from simtools.applications import docs_produce_production_summary


def test_application_initializes_model_reader():
    """Production summaries use the source-neutral model reader."""
    assert docs_produce_production_summary.APPLICATION.initialize_model_reader is True
