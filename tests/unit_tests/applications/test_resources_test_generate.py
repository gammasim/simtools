"""Tests for the resource-generation application."""

from simtools.applications import resources_test_generate


def test_model_source_arguments_are_registered_once():
    names = [argument.name for argument in resources_test_generate.APPLICATION.all_arguments]
    for name in (
        "simulation_models_path",
        "simulation_models_git_path",
        "simulation_models_git_revision",
    ):
        assert names.count(name) == 1
    resources_test_generate.APPLICATION.build_parser()


def test_application_does_not_initialize_model_reader():
    """Resource orchestration must not require an external model source during startup."""
    assert resources_test_generate.APPLICATION.initialize_model_reader is False
