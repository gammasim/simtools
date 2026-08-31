"""Tests for output-validator registration and dispatch."""

import json
from pathlib import Path

import pytest

from simtools.testing.output_validation import registry
from simtools.testing.output_validation.artifacts import OutputArtifact


def test_registry_dispatches_registered_validator(mocker):
    """Dispatch a rule to the validator registered for its type."""
    validator = mocker.Mock()
    mocker.patch.dict(registry.VALIDATORS, {"custom": validator})
    artifact = OutputArtifact(Path("output"), {})
    rule = {"type": "custom"}

    registry.run_validator(artifact, rule, {"configuration": {}})

    validator.assert_called_once_with(artifact, rule, {"configuration": {}})


def test_format_validator(tmp_test_directory):
    """Validate a configured output format."""
    output = Path(tmp_test_directory) / "result.json"
    output.write_text(json.dumps({"value": 1}), encoding="utf-8")
    registry.validate_format(OutputArtifact(output, {}), {"format": "json"}, {})
    with pytest.raises(AssertionError, match="not yaml"):
        registry.validate_format(OutputArtifact(output, {}), {"format": "yaml"}, {})


def test_log_validator_dispatches_expected_patterns(mocker):
    """Pass expected log patterns to the log assertion helper."""
    artifact = OutputArtifact(Path("output.log"), {})
    log_check = mocker.patch.object(registry.assertions, "check_log_files", return_value=True)

    registry.validate_log(artifact, {"expected": {"pattern": ["done"]}}, {})

    log_check.assert_called_once()


def test_log_validator_reports_failed_patterns(mocker):
    """Report a failed log-pattern check."""
    artifact = OutputArtifact(Path("output.log"), {})
    mocker.patch.object(registry.assertions, "check_log_files", return_value=False)

    with pytest.raises(AssertionError, match="failed pattern validation"):
        registry.validate_log(artifact, {}, {})


def test_registry_rejects_unknown_validator():
    """Reject a validation rule with an unknown type."""
    with pytest.raises(ValueError, match="Unknown output validator"):
        registry.run_validator(OutputArtifact(Path("output"), {}), {"type": "missing"}, {})


def test_registry_wraps_validator_errors_with_context(mocker):
    """Add output and validator context to validator failures."""
    mocker.patch.dict(registry.VALIDATORS, {"broken": mocker.Mock(side_effect=ValueError("bad"))})

    with pytest.raises(AssertionError, match=r"output.*broken.*bad"):
        registry.run_validator(OutputArtifact(Path("output"), {}), {"type": "broken"}, {})


def test_registry_reference_validator(mocker):
    """Resolve and compare a configured reference file."""
    artifact = OutputArtifact(Path("output.json"), {})
    reference_file = Path("reference.json")
    resolve = mocker.patch.object(registry.reference, "resolve_path", return_value=reference_file)
    compare = mocker.patch.object(registry.reference, "compare_files", return_value=True)

    registry.validate_reference(
        artifact,
        {"file": "reference.json", "tolerance": 0.2, "columns": ["value"], "metadata": True},
        {},
    )

    resolve.assert_called_once_with("reference.json")
    compare.assert_called_once_with(
        reference_file,
        artifact.path,
        0.2,
        ["value"],
        True,
        None,
        None,
    )


def test_registry_reference_validator_reports_difference(mocker):
    """Report a reference comparison failure."""
    artifact = OutputArtifact(Path("output.json"), {})
    reference_file = Path("reference.json")
    mocker.patch.object(registry.reference, "resolve_path", return_value=reference_file)
    mocker.patch.object(registry.reference, "compare_files", return_value=False)
    mocker.patch.object(registry.reference, "difference_report", return_value="- changed value")

    with pytest.raises(AssertionError, match="changed value"):
        registry.validate_reference(artifact, {"file": "reference.json"}, {})


def test_registry_delegates_table_and_schema_validators(mocker):
    """Delegate table, metadata, and schema validation to their modules."""
    artifact = OutputArtifact(Path("output.ecsv"), {})
    schema = mocker.patch.object(registry.table, "validate_data_schema")
    table = mocker.patch.object(registry.table, "validate_table")
    metadata = mocker.patch.object(registry.table, "validate_metadata")
    mocker.patch.object(registry.reference, "resolve_path", return_value=Path("schema.yml"))

    registry.validate_data_schema(artifact, {"schema": "schema.yml"}, {})
    registry.validate_table(artifact, {"minimum_rows": 1}, {})
    registry.validate_metadata(artifact, {"required_keys": ["summary"]}, {})

    schema.assert_called_once_with(artifact.path, Path("schema.yml"))
    table.assert_called_once_with(artifact.path, {"minimum_rows": 1})
    metadata.assert_called_once_with(artifact.path, {"required_keys": ["summary"]})


def test_registry_delegates_hdf5_validators(mocker):
    """Delegate HDF5 dataset and product validation."""
    artifact = OutputArtifact(Path("output.hdf5"), {})
    datasets = mocker.patch.object(registry.hdf5, "validate_datasets")
    product = mocker.patch.object(registry.hdf5, "validate_product")

    registry.validate_hdf5_datasets(
        artifact, {"required": ["DATA"], "minimum_rows": {"DATA": 1}}, {}
    )
    registry.validate_hdf5_product(artifact, {"product": "reduced_event_data"}, {})

    datasets.assert_called_once_with(artifact.path, required=["DATA"], minimum_rows={"DATA": 1})
    product.assert_called_once_with(artifact.path, "reduced_event_data")


def test_registry_delegates_simtel_and_model_parameter_validators(mocker):
    """Delegate sim_telarray and model-parameter validation."""
    artifact = OutputArtifact(Path("output"), {})
    simtel = mocker.patch.object(registry.simtel, "validate_output")
    model_parameter = mocker.patch.object(registry.model_parameters, "validate")
    config = {"configuration": {"model_version": "7.0.0"}}
    rule = {"type": "model_parameter"}

    registry.validate_simtel(artifact, {"event_type": "shower"}, {})
    registry.validate_model_parameter(artifact, rule, config)

    simtel.assert_called_once_with(artifact.path, {"event_type": "shower"})
    model_parameter.assert_called_once_with(artifact.path, rule, config["configuration"])


def test_registry_simtel_config_validator(mocker):
    """Resolve and compare a sim_telarray configuration reference."""
    artifact = OutputArtifact(Path("output.cfg"), {})
    reference_file = Path("reference.cfg")
    mocker.patch.object(registry.reference, "resolve_path", return_value=reference_file)
    compare = mocker.patch.object(registry.simtel, "compare_config_files", return_value=True)

    registry.validate_simtel_config(artifact, {"reference": "reference.cfg"}, {})

    compare.assert_called_once_with(reference_file, artifact.path)


def test_registry_simtel_config_validator_reports_difference(mocker):
    """Report a sim_telarray configuration comparison failure."""
    artifact = OutputArtifact(Path("output.cfg"), {})
    reference_file = Path("reference.cfg")
    mocker.patch.object(registry.reference, "resolve_path", return_value=reference_file)
    mocker.patch.object(registry.simtel, "compare_config_files", return_value=False)

    with pytest.raises(AssertionError, match="differs from sim_telarray reference"):
        registry.validate_simtel_config(artifact, {"reference": "reference.cfg"}, {})
