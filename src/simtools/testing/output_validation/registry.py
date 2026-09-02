"""Validator registry and handlers for integration-test outputs."""

from simtools.testing import assertions
from simtools.testing.output_validation import hdf5, model_parameters, reference, simtel, table


def validate_format(artifact, rule, _context):
    """Validate that an output has the configured format."""
    if not assertions.assert_file_type(rule["format"], artifact.path):
        raise AssertionError(f"Output '{artifact.path}' is not {rule['format']} format.")


def validate_reference(artifact, rule, _context):
    """Compare an artifact with a reference file."""
    reference_file = reference.resolve_path(rule["file"])
    if not reference.compare_files(
        reference_file,
        artifact.path,
        rule.get("tolerance", 1.0e-5),
        rule.get("columns"),
        rule.get("metadata", False),
        rule.get("filters"),
        rule.get("key_columns"),
    ):
        raise AssertionError(
            f"Output '{artifact.path}' differs from reference '{reference_file}'.\n"
            f"{reference.difference_report(reference_file, artifact.path)}"
        )


def validate_data_schema(artifact, rule, _context):
    """Validate a tabular output against a simtools data schema."""
    table.validate_data_schema(artifact.path, reference.resolve_path(rule["schema"]))


def validate_table(artifact, rule, _context):
    """Validate table rows, uniqueness, allowed values, and ranges."""
    table.validate_table(artifact.path, rule)


def validate_metadata(artifact, rule, _context):
    """Validate ECSV metadata keys and content relations."""
    table.validate_metadata(artifact.path, rule)


def validate_hdf5_datasets(artifact, rule, _context):
    """Validate required HDF5 datasets and minimum row counts."""
    hdf5.validate_datasets(
        artifact.path,
        required=rule.get("required"),
        minimum_rows=rule.get("minimum_rows"),
    )


def validate_hdf5_product(artifact, rule, _context):
    """Validate a registered structured HDF5 product."""
    hdf5.validate_product(artifact.path, rule["product"])


def validate_log(artifact, rule, _context):
    """Validate wanted and forbidden log patterns."""
    if not assertions.check_log_files(
        artifact.path, {"expected_log_output": rule.get("expected", {})}
    ):
        raise AssertionError(f"Output log '{artifact.path}' failed pattern validation.")


def validate_simtel(artifact, rule, _context):
    """Validate sim_telarray event output and metadata."""
    simtel.validate_output(artifact.path, rule)


def validate_simtel_config(artifact, rule, _context):
    """Compare a generated sim_telarray configuration with a reference."""
    reference_file = reference.resolve_path(rule["reference"])
    if not simtel.compare_config_files(reference_file, artifact.path):
        raise AssertionError(
            f"Output '{artifact.path}' differs from sim_telarray reference '{reference_file}'."
        )


def validate_model_parameter(artifact, rule, context):
    """Compare a generated model parameter with its database value."""
    model_parameters.validate(artifact.path, rule, context["configuration"])


VALIDATORS = {
    "format": validate_format,
    "reference": validate_reference,
    "data_schema": validate_data_schema,
    "table": validate_table,
    "metadata": validate_metadata,
    "hdf5_datasets": validate_hdf5_datasets,
    "hdf5_product": validate_hdf5_product,
    "log": validate_log,
    "simtel": validate_simtel,
    "simtel_config": validate_simtel_config,
    "model_parameter": validate_model_parameter,
}


def run_validator(artifact, rule, context):
    """Run one explicitly configured output validator."""
    try:
        validator = VALIDATORS[rule["type"]]
    except KeyError as exc:
        raise ValueError(f"Unknown output validator '{rule.get('type')}'.") from exc
    try:
        validator(artifact, rule, context)
    except Exception as exc:
        raise AssertionError(
            f"Output '{artifact.path}' failed validator '{rule['type']}': {exc}"
        ) from exc
