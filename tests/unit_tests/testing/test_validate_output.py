"""Tests for composable integration-output validation."""

import json
from pathlib import Path

import pytest
import yaml
from astropy import units as u
from astropy.table import Table

from simtools.testing import validate_output
from simtools.testing.output_validation import (
    hdf5,
    model_parameters,
    reference,
    registry,
    runner,
    simtel,
    table,
)
from simtools.testing.output_validation.artifacts import OutputArtifact


def _write_table(path, metadata=None):
    output = Table({"id": [1, 2], "value": [1.0, 2.0], "label": ["a", "b"]})
    output["value"].unit = u.m
    output.meta = metadata or {"summary": {"rows": 2, "total": 3.0}}
    output.write(path, format="ascii.ecsv", overwrite=True)


def _write_schema(path):
    schema = {
        "schema_version": "0.1.0",
        "data": [
            {
                "type": "data_table",
                "table_columns": [
                    {"name": "id", "required": True, "type": "int64"},
                    {"name": "value", "required": True, "type": "float64", "unit": "m"},
                    {"name": "label", "required": True, "type": "string"},
                ],
            }
        ],
    }
    path.write_text(yaml.safe_dump(schema), encoding="utf-8")


def test_output_artifact_resolves_and_checks_existence(tmp_test_directory):
    output = Path(tmp_test_directory) / "nested" / "result.txt"
    output.parent.mkdir()
    output.touch()
    artifact = OutputArtifact.from_descriptor(
        {"output_path": tmp_test_directory},
        {"path_descriptor": "output_path", "output_sub_path": "nested", "file": output.name},
    )
    artifact.assert_exists()
    assert artifact.path == output


def test_output_artifact_reports_configuration_and_missing_file(tmp_test_directory):
    with pytest.raises(KeyError, match="Path missing"):
        OutputArtifact.from_descriptor(
            {"output_path": tmp_test_directory},
            {"path_descriptor": "missing", "file": "result.txt"},
        )
    artifact = OutputArtifact(Path(tmp_test_directory) / "missing.txt", {})
    with pytest.raises(AssertionError, match="does not exist"):
        artifact.assert_exists()


def test_runner_dispatches_validations_in_order(tmp_test_directory, mocker):
    output = Path(tmp_test_directory) / "result.txt"
    output.touch()
    dispatch = mocker.patch("simtools.testing.output_validation.runner.run_validator")
    config = {
        "configuration": {"output_path": tmp_test_directory},
        "integration_tests": [
            {
                "test_outputs": [
                    {
                        "path_descriptor": "output_path",
                        "file": output.name,
                        "validations": [{"type": "format", "format": "txt"}, {"type": "log"}],
                    }
                ]
            }
        ],
    }
    validate_output.validate_application_output(config)
    assert [call.args[1]["type"] for call in dispatch.call_args_list] == ["format", "log"]


def test_runner_applies_model_version_filters(tmp_test_directory, mocker):
    output = Path(tmp_test_directory) / "result.txt"
    output.touch()
    dispatch = mocker.patch("simtools.testing.output_validation.runner.run_validator")
    config = {
        "configuration": {"output_path": tmp_test_directory},
        "integration_tests": [
            {
                "test_outputs": [
                    {
                        "path_descriptor": "output_path",
                        "file": output.name,
                        "model_versions": ["7.0.0"],
                        "validations": [{"type": "format", "format": "txt"}],
                    }
                ]
            }
        ],
    }
    runner.validate_application_output(config, from_config_file="6.0.0")
    dispatch.assert_not_called()
    runner.validate_application_output(config, from_config_file="7.0.0")
    dispatch.assert_called_once()


def test_format_validator(tmp_test_directory):
    output = Path(tmp_test_directory) / "result.json"
    output.write_text(json.dumps({"value": 1}), encoding="utf-8")
    registry.validate_format(OutputArtifact(output, {}), {"format": "json"}, {})
    with pytest.raises(AssertionError, match="not yaml"):
        registry.validate_format(OutputArtifact(output, {}), {"format": "yaml"}, {})


def test_reference_comparison_all_column_types_and_metadata(tmp_test_directory):
    first = Path(tmp_test_directory) / "first.ecsv"
    second = Path(tmp_test_directory) / "second.ecsv"
    _write_table(first)
    _write_table(second)
    assert reference.compare_files(first, second, metadata=True)
    changed = Table.read(second, format="ascii.ecsv")
    changed["label"][1] = "x"
    changed.write(second, format="ascii.ecsv", overwrite=True)
    assert not reference.compare_files(first, second)


def test_reference_comparison_typed_filters_and_key_order(tmp_test_directory):
    first = Path(tmp_test_directory) / "first.ecsv"
    second = Path(tmp_test_directory) / "second.ecsv"
    Table({"id": [2, 1, 3], "group": ["keep", "keep", "drop"], "value": [2.0, 1.0, 9.0]}).write(
        first, format="ascii.ecsv"
    )
    Table({"id": [1, 2, 4], "group": ["keep", "keep", "drop"], "value": [1.0, 2.0, 8.0]}).write(
        second, format="ascii.ecsv"
    )

    assert reference.compare_files(
        first,
        second,
        filters=[{"column": "group", "operator": "equal", "value": "keep"}],
        key_columns=["id"],
    )


def test_reference_comparison_rejects_duplicate_keys(tmp_test_directory):
    first = Path(tmp_test_directory) / "first.ecsv"
    second = Path(tmp_test_directory) / "second.ecsv"
    Table({"id": [1, 1]}).write(first, format="ascii.ecsv")
    Table({"id": [1, 1]}).write(second, format="ascii.ecsv")

    with pytest.raises(ValueError, match="not unique"):
        reference.compare_files(first, second, key_columns=["id"])


def test_json_reference_comparison_ignores_schema_version(tmp_test_directory):
    first = Path(tmp_test_directory) / "first.json"
    second = Path(tmp_test_directory) / "second.json"
    first.write_text(json.dumps({"schema_version": "1", "value": [1.0]}), encoding="utf-8")
    second.write_text(json.dumps({"schema_version": "2", "value": [1.0]}), encoding="utf-8")
    assert reference.compare_json_or_yaml_files(first, second)


def test_table_and_metadata_validators(tmp_test_directory):
    output = Path(tmp_test_directory) / "table.ecsv"
    _write_table(output)
    table.validate_table(
        output,
        {
            "minimum_rows": 2,
            "unique_columns": ["id"],
            "columns": {
                "label": {"allowed_values": ["a", "b"]},
                "value": {"range": {"minimum": 1.0, "maximum": 2.0, "unit": "m"}},
            },
        },
    )
    table.validate_metadata(
        output,
        {
            "required_keys": ["summary"],
            "relations": [
                {"left": "summary.rows", "equals": "table.row_count"},
                {"left": "summary.total", "equals": "table.column_sum", "column": "value"},
            ],
        },
    )


@pytest.mark.parametrize(
    ("rule", "message"),
    [
        ({"minimum_rows": 3}, "rows"),
        ({"columns": {"label": {"allowed_values": ["x"]}}}, "outside"),
        ({"columns": {"value": {"range": {"minimum": 3.0, "unit": "m"}}}}, "minimum"),
    ],
)
def test_table_validator_failures(tmp_test_directory, rule, message):
    output = Path(tmp_test_directory) / "table.ecsv"
    _write_table(output)
    with pytest.raises(AssertionError, match=message):
        table.validate_table(output, rule)


def test_metadata_validator_failures(tmp_test_directory):
    output = Path(tmp_test_directory) / "table.ecsv"
    _write_table(output)
    with pytest.raises(AssertionError, match="no metadata key"):
        table.validate_metadata(output, {"required_keys": ["missing"]})
    with pytest.raises(AssertionError, match="expected"):
        table.validate_metadata(
            output,
            {"relations": [{"left": "summary.total", "equals": "table.row_count"}]},
        )


def test_data_schema_validator(tmp_test_directory):
    output = Path(tmp_test_directory) / "table.ecsv"
    schema_file = Path(tmp_test_directory) / "table.schema.yml"
    _write_table(output)
    _write_schema(schema_file)
    table.validate_data_schema(output, schema_file)
    Table({"other": [1]}).write(output, format="ascii.ecsv", overwrite=True)
    with pytest.raises(AssertionError, match="data-product schema"):
        table.validate_data_schema(output, schema_file)


def test_hdf5_validators_dispatch(mocker):
    path = Path("output.hdf5")
    required = mocker.patch.object(hdf5.assertions, "assert_hdf5_datasets")
    minimum = mocker.patch.object(hdf5.assertions, "assert_hdf5_dataset_min_rows")
    hdf5.validate_datasets(path, required=["DATA"], minimum_rows={"DATA": 1})
    required.assert_called_once_with(path, ["DATA"])
    minimum.assert_called_once_with(path, {"DATA": 1})
    product = mocker.patch(
        "simtools.testing.output_validation.hdf5.output_validator.validate_reduced_event_data_file"
    )
    hdf5.validate_product(path, "reduced_event_data")
    product.assert_called_once_with(path)
    with pytest.raises(ValueError, match="Unsupported HDF5 product"):
        hdf5.validate_product(path, "missing")


def test_log_and_simtel_validators(mocker):
    artifact = OutputArtifact(Path("output.log"), {})
    log_check = mocker.patch.object(registry.assertions, "check_log_files", return_value=True)
    registry.validate_log(artifact, {"expected": {"pattern": ["done"]}}, {})
    log_check.assert_called_once()
    simtel_check = mocker.patch.object(
        simtel.assertions, "check_output_from_sim_telarray", return_value=True
    )
    simtel.validate_output(
        Path("output.simtel.zst"),
        {"event_type": "shower", "event": {"pe_sum": {"range": [20, 1000]}}},
    )
    simtel_check.assert_called_once_with(
        Path("output.simtel.zst"),
        {
            "expected_sim_telarray_output": {
                "event_type": "shower",
                "pe_sum": [20, 1000],
            }
        },
    )


def test_simtel_config_validator(tmp_test_directory, mocker):
    reference_file = Path(tmp_test_directory) / "reference.cfg"
    output = Path(tmp_test_directory) / "output.cfg"
    reference_file.write_text("parameter = 1\n", encoding="utf-8")
    output.write_text("parameter = 1\n", encoding="utf-8")
    mocker.patch.object(simtel, "_assignment_metadata_keys", return_value=set())
    assert simtel.compare_config_files(reference_file, output)
    output.write_text("parameter = 2\n", encoding="utf-8")
    assert not simtel.compare_config_files(reference_file, output)


def test_model_parameter_validator(tmp_test_directory, mocker):
    output = Path(tmp_test_directory) / "parameter.json"
    output.write_text(json.dumps({"value": [1.0, 2.0]}), encoding="utf-8")
    database = mocker.patch.object(model_parameters.db_handler, "DatabaseHandler")
    database.return_value.get_model_parameter.return_value = {"parameter": {"value": [2.0, 4.0]}}
    model_parameters.validate(
        output,
        {
            "reference_parameter_name": "parameter",
            "tolerance": 1.0e-5,
            "scaling": 2.0,
        },
        {"site": "North", "telescope": "LSTN-01", "model_version": "7.0.0"},
    )


def test_registry_rejects_unknown_validator():
    with pytest.raises(ValueError, match="Unknown output validator"):
        registry.run_validator(OutputArtifact(Path("output"), {}), {"type": "missing"}, {})


def test_registry_wraps_validator_errors_with_context(mocker):
    mocker.patch.dict(registry.VALIDATORS, {"broken": mocker.Mock(side_effect=ValueError("bad"))})

    with pytest.raises(AssertionError, match=r"output.*broken.*bad"):
        registry.run_validator(OutputArtifact(Path("output"), {}), {"type": "broken"}, {})
