"""Orchestration of integration-test output validation."""

from simtools.application.model_reader import create_model_reader_from_configuration
from simtools.testing.output_validation.artifacts import OutputArtifact
from simtools.testing.output_validation.registry import run_validator


def versions_match(from_command_line, from_config_file):
    """Return whether configured output validations apply to model versions."""
    if from_command_line is None:
        return True
    command_versions = (
        from_command_line if isinstance(from_command_line, list) else [from_command_line]
    )
    config_versions = from_config_file if isinstance(from_config_file, list) else [from_config_file]
    return bool(set(command_versions) & set(config_versions))


def validate_application_output(
    config, from_command_line=None, from_config_file=None, model_reader=None
):
    """Validate all explicitly declared output artifacts.

    Parameters
    ----------
    model_reader : object, optional
        Reader to reuse for model-parameter validations.
    """
    if not versions_match(from_command_line, from_config_file):
        return
    configuration = config.get("configuration")
    if configuration is None:
        return
    active_versions = from_command_line or from_config_file
    context = {
        "configuration": configuration,
        "model_reader": model_reader,
    }
    for integration_test in config.get("integration_tests", []):
        for descriptor in integration_test.get("test_outputs", []):
            _validate_descriptor(descriptor, configuration, active_versions, context)


def _validate_descriptor(descriptor, configuration, active_versions, context):
    """Validate one output descriptor when its model version applies."""
    descriptor_versions = descriptor.get("model_versions")
    if descriptor_versions and not versions_match(active_versions, descriptor_versions):
        return
    artifact = OutputArtifact.from_descriptor(configuration, descriptor)
    artifact.assert_exists()
    for rule in descriptor.get("validations", []):
        _validate_rule(artifact, rule, configuration, context)


def _validate_rule(artifact, rule, configuration, context):
    """Run one validation rule, creating a shared model reader if needed."""
    if rule["type"] == "model_parameter" and context["model_reader"] is None:
        context["model_reader"] = _create_model_reader(configuration)
    run_validator(artifact, rule, context)


def _create_model_reader(configuration):
    """Create a reader for model-parameter output validation."""
    return create_model_reader_from_configuration(configuration)
