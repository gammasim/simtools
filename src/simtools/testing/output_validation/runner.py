"""Orchestration of integration-test output validation."""

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


def validate_application_output(config, from_command_line=None, from_config_file=None):
    """Validate all explicitly declared output artifacts."""
    if not versions_match(from_command_line, from_config_file):
        return
    active_versions = from_command_line or from_config_file
    for integration_test in config.get("integration_tests", []):
        for descriptor in integration_test.get("test_outputs", []):
            descriptor_versions = descriptor.get("model_versions")
            if descriptor_versions and not versions_match(active_versions, descriptor_versions):
                continue
            artifact = OutputArtifact.from_descriptor(config["configuration"], descriptor)
            artifact.assert_exists()
            for rule in descriptor.get("validations", []):
                run_validator(artifact, rule, config)
