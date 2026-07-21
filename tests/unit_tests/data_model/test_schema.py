#!/usr/bin/python3

import logging
from pathlib import Path

import jsonschema
import pytest
import yaml
from astropy.table import Table
from packaging.specifiers import InvalidSpecifier

from simtools.constants import (
    MODEL_PARAMETER_DESCRIPTION_METASCHEMA,
    MODEL_PARAMETER_METASCHEMA,
    MODEL_PARAMETER_SCHEMA_PATH,
    SCHEMA_PATH,
    SIM_TELARRAY_META_PARAMETER_METASCHEMA,
    SIM_TELARRAY_META_PARAMETER_REGISTRY,
)
from simtools.data_model import schema, schema_loader
from simtools.io import ascii_handler
from simtools.utils import names


def test_get_model_parameter_schema_files(tmp_test_directory):
    par, files = schema.get_model_parameter_schema_files()
    assert len(files)
    assert files[0].is_file()
    assert "num_gains" in par

    # no files in the directory
    with pytest.raises(FileNotFoundError, match=r"^No schema files"):
        schema.get_model_parameter_schema_files(tmp_test_directory)

    # directory does not exist
    with pytest.raises(FileNotFoundError, match=r"^No schema files"):
        schema.get_model_parameter_schema_files("not_a_directory")


def test_get_model_parameter_schema_file():
    schema_file = str(schema.get_model_parameter_schema_file("num_gains"))

    assert str(MODEL_PARAMETER_SCHEMA_PATH / "num_gains.schema.yml") in schema_file

    with pytest.raises(FileNotFoundError, match=r"^Schema file not found:"):
        schema.get_model_parameter_schema_file("not_a_parameter")


def test_get_model_parameter_schema_returns_cached_schema():
    schema.load_schema.cache_clear()
    schema_loader.load_schema.cache_clear()
    schema_1 = schema.get_model_parameter_schema("mirror_focal_length", "0.1.0")
    schema_2 = schema.get_model_parameter_schema("mirror_focal_length", "0.1.0")

    assert schema_1 is schema_2


def test_load_schema_cache_separates_versions(mocker):
    schema.load_schema.cache_clear()
    schema_loader.load_schema.cache_clear()
    collect_data = mocker.spy(ascii_handler, "collect_data_from_file")

    schema_1 = schema.load_schema(MODEL_PARAMETER_METASCHEMA, "0.1.0")
    schema_2 = schema.load_schema(MODEL_PARAMETER_METASCHEMA, "0.2.0")
    schema_1_cached = schema.load_schema(MODEL_PARAMETER_METASCHEMA, "0.1.0")

    assert schema_1 is schema_1_cached
    assert schema_1 is not schema_2
    assert collect_data.call_count == 2


def test_model_parameter_cache_is_shared_with_names(mocker):
    schema.load_schema.cache_clear()
    schema_loader.load_schema.cache_clear()
    names._load_model_parameters.cache_clear()
    collect_data = mocker.spy(ascii_handler, "collect_data_from_file")

    model_parameters = names.model_parameters()
    schema_file = schema.get_model_parameter_schema_file("mirror_focal_length")
    reads_before = sum(
        call.kwargs.get("file_name") == schema_file for call in collect_data.call_args_list
    )
    loaded_schema = schema.get_model_parameter_schema("mirror_focal_length")
    reads_after = sum(
        call.kwargs.get("file_name") == schema_file for call in collect_data.call_args_list
    )

    assert loaded_schema == model_parameters["mirror_focal_length"]
    assert loaded_schema is not model_parameters["mirror_focal_length"]
    assert reads_before == reads_after == 1


def test_validate_sim_telarray_meta_parameter_registry_schema():
    registry = ascii_handler.collect_data_from_file(SIM_TELARRAY_META_PARAMETER_REGISTRY)

    schema.validate_dict_using_schema(
        registry,
        schema_file=SIM_TELARRAY_META_PARAMETER_METASCHEMA,
        offline=True,
        ignore_software_version=True,
    )

    assert "generated_meta_parameters" in registry
    assert "model_parameters" not in registry


def test_get_parameter_type_and_unit_from_schema():
    assert (
        schema.get_parameter_attribute_from_schema("mirror_focal_length", "0.1.0", "type")
        == "float64"
    )
    assert (
        schema.get_parameter_attribute_from_schema("mirror_focal_length", "0.1.0", "unit") == "cm"
    )

    assert schema.get_parameter_attribute_from_schema("flasher_pulse_shape", "0.2.0", "type") == [
        "string",
        "float64",
        "float64",
    ]
    assert schema.get_parameter_attribute_from_schema("flasher_pulse_shape", "0.2.0", "unit") == [
        None,
        "ns",
        "ns",
    ]


def testget_parameter_attribute_from_schema_with_dict_data(mocker):
    """Test helper handles schema entries where data is represented as a dict."""
    mocker.patch(
        "simtools.data_model.schema.get_model_parameter_schema",
        return_value={"data": {"type": "float64", "unit": "dimensionless"}},
    )

    assert schema.get_parameter_attribute_from_schema("dummy", "0.1.0", "type") == "float64"
    assert schema.get_parameter_attribute_from_schema("dummy", "0.1.0", "unit") is None


def testget_parameter_attribute_from_schema_with_invalid_data_type(mocker):
    """Test helper returns None for unsupported data structures."""
    mocker.patch(
        "simtools.data_model.schema.get_model_parameter_schema",
        return_value={"data": "invalid"},
    )

    assert schema.get_parameter_attribute_from_schema("dummy", "0.1.0", "type") is None


def test_get_model_parameter_schema_version():
    most_recent = schema.get_model_parameter_schema_version()
    assert most_recent == "0.3.0"

    assert schema.get_model_parameter_schema_version("0.2.0") == "0.2.0"
    assert schema.get_model_parameter_schema_version("0.1.0") == "0.1.0"

    with pytest.raises(ValueError, match=r"^Schema version 0.0.1 not found in"):
        schema.get_model_parameter_schema_version("0.0.1")


def test_validate_dict_using_schema(tmp_test_directory, caplog):
    with caplog.at_level(logging.WARNING):
        schema.validate_dict_using_schema(None, None)
    assert "No schema provided for validation of" in caplog.text

    sample_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
        "meta_schema_url": "string",
        "required": ["name", "age"],
    }

    schema_file = Path(tmp_test_directory) / "schema.yml"
    with open(schema_file, "w", encoding="utf-8") as f:
        yaml.dump(sample_schema, f)

    # sample data dictionary to be validated
    data = {"name": "John", "age": 30}

    schema.validate_dict_using_schema(data, schema_file, offline=True)

    invalid_data = {"name": "Alice", "age": "Thirty"}
    with pytest.raises(jsonschema.exceptions.ValidationError):
        schema.validate_dict_using_schema(invalid_data, schema_file)


def test_runtime_environment_definition_is_reused_by_workflow_schema():
    workflow_config = {
        "schema_version": "0.4.0",
        "schema_name": "application_workflow.metaschema",
        "runtime_environment": {
            "container_engine": "podman",
            "image": "test-image",
            "network": "simtools-mongo-network",
            "environment_file": ".env",
            "options": ["--arch amd64"],
        },
        "applications": [
            {
                "application": "simtools-test",
                "configuration": {},
            }
        ],
    }

    schema.validate_dict_using_schema(
        workflow_config, schema_file=SCHEMA_PATH / "application_workflow.metaschema.yml"
    )

    workflow_config["runtime_environment"]["unknown"] = "value"
    with pytest.raises(jsonschema.ValidationError):
        schema.validate_dict_using_schema(
            workflow_config, schema_file=SCHEMA_PATH / "application_workflow.metaschema.yml"
        )


def test_application_workflow_schema_accepts_optional_docs_metadata():
    workflow_config = {
        "schema_version": "0.4.0",
        "schema_name": "application_workflow.metaschema",
        "applications": [
            {
                "application": "simtools-test",
                "docs": {
                    "title": "Example title",
                    "summary": "Example summary.",
                },
                "configuration": {},
            }
        ],
    }

    schema.validate_dict_using_schema(
        workflow_config,
        schema_file=SCHEMA_PATH / "application_workflow.metaschema.yml",
    )


def test_application_workflow_schema_accepts_expected_failure_reason():
    """Allow workflows to document expected application failures."""
    workflow_config = {
        "schema_version": "0.4.0",
        "schema_name": "application_workflow.metaschema",
        "applications": [
            {
                "application": "simtools-test",
                "configuration": {},
                "xfail": "known issue",
            }
        ],
    }

    schema.validate_dict_using_schema(
        workflow_config,
        schema_file=SCHEMA_PATH / "application_workflow.metaschema.yml",
    )


def _output_validation_workflow(*rules):
    """Build a minimal workflow containing declarative output rules."""
    return {
        "schema_version": "0.4.0",
        "schema_name": "application_workflow.metaschema",
        "applications": [
            {
                "application": "simtools-test",
                "configuration": {"output_path": "output"},
                "integration_tests": [{"output_validation": list(rules)}],
            }
        ],
    }


def _valid_output_validation_rule():
    """Build a representative table rule for metaschema tests."""
    return {
        "name": "table",
        "path_descriptor": "output_path",
        "file": "output.ecsv",
        "data_product_schema": "schema.yml",
        "minimum_rows": 1,
        "unique_columns": ["id"],
        "columns": {
            "energy": {
                "range": {"minimum": 1.0, "unit": "GeV"},
            },
        },
        "metadata": {
            "required_keys": ["summary"],
            "row_count": "summary.rows",
            "column_sums": {"energy": "summary.total"},
        },
    }


def test_application_workflow_schema_accepts_output_validation_rules():
    """Test the table output-validation configuration shape."""
    workflow_config = _output_validation_workflow(_valid_output_validation_rule())

    schema.validate_dict_using_schema(
        workflow_config,
        schema_file=SCHEMA_PATH / "application_workflow.metaschema.yml",
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda rule: rule.update({"unknown": True}),
        lambda rule: rule.update({"minimum_rows": -1}),
        lambda rule: rule.update({"unique_columns": ["id", "id"]}),
        lambda rule: rule["columns"]["energy"].update({"range": {"minimum": "bad"}}),
        lambda rule: rule["columns"]["energy"].update({"range": {"unit": "GeV"}}),
        lambda rule: rule["columns"].update({"id": {}}),
        lambda rule: rule["metadata"].update({"unknown": "value"}),
    ],
)
def test_application_workflow_schema_rejects_malformed_output_validation(change):
    """Reject unknown properties and malformed declarative validation rules."""
    rule = _valid_output_validation_rule()
    change(rule)
    workflow_config = _output_validation_workflow(rule)

    with pytest.raises(jsonschema.ValidationError):
        schema.validate_dict_using_schema(
            workflow_config,
            schema_file=SCHEMA_PATH / "application_workflow.metaschema.yml",
        )


def test_validate_dict_using_schema_remote(tmp_test_directory, mocker):
    sample_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
        "meta_schema_url": "string",
        "required": ["name", "age"],
    }

    schema_file = Path(tmp_test_directory) / "schema.yml"
    with open(schema_file, "w", encoding="utf-8") as f:
        yaml.dump(sample_schema, f)

    # sample data dictionary to be validated
    data = {"name": "John", "age": 30}

    mock_url_exists = mocker.patch("simtools.data_model.schema.gen.url_exists")

    # with valid meta_schema_url
    mock_url_exists.return_value = True
    data["meta_schema_url"] = "https://github.com/gammasim/simtools"
    schema.validate_dict_using_schema(data, schema_file)
    mock_url_exists.assert_called_with("https://github.com/gammasim/simtools")

    mock_url_exists.return_value = False
    data["meta_schema_url"] = "https://invalid_url"
    with pytest.raises(FileNotFoundError, match=r"^Meta schema URL does not exist:"):
        schema.validate_dict_using_schema(data, schema_file)
    mock_url_exists.assert_called_with("https://invalid_url")


def test_validate_schema_astropy_units(caplog):
    success_string = "Successful validation of data using schema"

    _dict_1 = ascii_handler.collect_data_from_file(
        file_name=MODEL_PARAMETER_SCHEMA_PATH / "num_gains.schema.yml"
    )
    with caplog.at_level(logging.DEBUG):
        schema.validate_dict_using_schema(
            data=_dict_1, schema_file=MODEL_PARAMETER_DESCRIPTION_METASCHEMA, offline=True
        )
    assert success_string in caplog.text

    # m and cm
    _dict_1["data"][0]["unit"] = "m"
    with caplog.at_level(logging.DEBUG):
        schema.validate_dict_using_schema(
            data=_dict_1, schema_file=MODEL_PARAMETER_DESCRIPTION_METASCHEMA, offline=True
        )
    assert success_string in caplog.text
    _dict_1["data"][0]["unit"] = "cm"
    with caplog.at_level(logging.DEBUG):
        schema.validate_dict_using_schema(
            data=_dict_1, schema_file=MODEL_PARAMETER_DESCRIPTION_METASCHEMA, offline=True
        )
    assert success_string in caplog.text

    # combined units
    _dict_1["data"][0]["unit"] = "cm/s"
    with caplog.at_level(logging.DEBUG):
        schema.validate_dict_using_schema(
            data=_dict_1, schema_file=MODEL_PARAMETER_DESCRIPTION_METASCHEMA, offline=True
        )
    assert success_string in caplog.text
    _dict_1["data"][0]["unit"] = "km/ s"
    with caplog.at_level(logging.DEBUG):
        schema.validate_dict_using_schema(
            data=_dict_1, schema_file=MODEL_PARAMETER_DESCRIPTION_METASCHEMA, offline=True
        )
    assert success_string in caplog.text

    # dimensionless
    _dict_1["data"][0]["unit"] = "dimensionless"
    with caplog.at_level(logging.DEBUG):
        schema.validate_dict_using_schema(
            data=_dict_1, schema_file=MODEL_PARAMETER_DESCRIPTION_METASCHEMA, offline=True
        )
    assert success_string in caplog.text
    _dict_1["data"][0]["unit"] = ""
    with caplog.at_level(logging.DEBUG):
        schema.validate_dict_using_schema(
            data=_dict_1, schema_file=MODEL_PARAMETER_DESCRIPTION_METASCHEMA, offline=True
        )
    assert success_string in caplog.text

    # not good
    _dict_1["data"][0]["unit"] = "not_a_unit"
    with pytest.raises(ValueError, match="'not_a_unit' is not a valid Unit"):
        schema.validate_dict_using_schema(
            data=_dict_1, schema_file=MODEL_PARAMETER_DESCRIPTION_METASCHEMA, offline=True
        )


@pytest.mark.parametrize("model_status", ["development", "production", "superseded"])
def test_validate_simulation_models_info_schema_accepts_model_status(model_status):
    """Test simulation models info schema accepts all supported model_status values."""
    data = {
        "schema_version": "0.2.0",
        "model_version": "6.1.0",
        "model_update": "patch_update",
        "model_version_history": ["6.0.2"],
        "model_status": model_status,
        "description": "test",
        "changes": {},
    }

    schema.validate_dict_using_schema(
        data=data,
        schema_file="simulation_models_info.schema.yml",
        offline=True,
    )


def test_validate_simulation_models_info_schema_rejects_invalid_model_status():
    """Test simulation models info schema rejects unsupported model_status values."""
    data = {
        "schema_version": "0.2.0",
        "model_version": "6.1.0",
        "model_update": "patch_update",
        "model_version_history": ["6.0.2"],
        "model_status": "ready-for-production",
        "description": "test",
        "changes": {},
    }

    with pytest.raises(jsonschema.exceptions.ValidationError):
        schema.validate_dict_using_schema(
            data=data,
            schema_file="simulation_models_info.schema.yml",
            offline=True,
        )


def test_validate_simulation_models_info_schema_allows_missing_model_status_for_010():
    """Test simulation models info schema 0.1.0 keeps backward compatibility."""
    data = {
        "schema_version": "0.1.0",
        "model_version": "6.1.0",
        "model_update": "patch_update",
        "model_version_history": ["6.0.2"],
        "description": "test",
        "changes": {},
    }

    schema.validate_dict_using_schema(
        data=data,
        schema_file="simulation_models_info.schema.yml",
        offline=True,
    )


def test_load_schema(caplog, tmp_test_directory):
    _metadata_schema = schema.load_schema()
    assert isinstance(_metadata_schema, dict)
    assert len(_metadata_schema) > 0

    with pytest.raises(FileNotFoundError):
        schema.load_schema(schema_file="not_existing_file")

    _schema_1 = schema.load_schema(MODEL_PARAMETER_METASCHEMA, "0.1.0")
    assert _schema_1["schema_version"] == "0.1.0"
    _schema_2 = schema.load_schema(MODEL_PARAMETER_METASCHEMA, "0.2.0")
    assert _schema_2["schema_version"] == "0.2.0"

    # test a single doc yaml file (write a temporary schema file; to make sure it is a single doc)
    tmp_schema_file = Path(tmp_test_directory) / "schema.yml"
    with open(tmp_schema_file, "w", encoding="utf-8") as f:
        yaml.dump(_schema_2, f)

    with caplog.at_level(logging.WARNING):
        schema.load_schema(tmp_schema_file, "0.3.0")
    assert "Schema version 0.3.0 does not match 0.2.0" in caplog.text


def test_add_array_elements():
    test_dict_1 = {"data": {"InstrumentTypeElement": {"enum": ["LSTN", "MSTN"]}}}
    test_dict_added = schema._add_array_elements("InstrumentTypeElement", test_dict_1)
    assert len(test_dict_added["data"]["InstrumentTypeElement"]["enum"]) > 2
    test_dict_2 = {"data": {"InstrumentTypeElement": {"not_the_right_enum": ["LSTN", "MSTN"]}}}
    test_dict_added_2 = schema._add_array_elements("InstrumentTypeElement", test_dict_2)
    assert len(test_dict_added_2["data"]["InstrumentTypeElement"]["enum"]) > 2


def test_retrieve_yaml_schema_from_uri(tmp_path, monkeypatch):
    # Create a dummy schema file
    dummy_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"foo": {"type": "string"}},
    }
    schema_file = tmp_path / "dummy.schema.yml"
    with open(schema_file, "w", encoding="utf-8") as f:
        yaml.dump(dummy_schema, f)

    # Patch SCHEMA_PATH to tmp_path for this test
    monkeypatch.setattr(schema, "SCHEMA_PATH", tmp_path)

    # The uri should be 'file:/dummy.schema.yml'
    uri = f"file:/{schema_file.name}"

    resource = schema._retrieve_yaml_schema_from_uri(uri)
    assert hasattr(resource, "contents")
    assert resource.contents["type"] == "object"
    assert "foo" in resource.contents["properties"]

    # Test with non-existing file
    bad_uri = "file:/not_existing_file.schema.yml"
    with pytest.raises(FileNotFoundError):
        schema._retrieve_yaml_schema_from_uri(bad_uri)


def test_get_schema_version_from_data_with_schema_version():
    data = {"schema_version": "1.2.3"}
    result = schema.get_schema_version_from_data(data)
    assert result == "1.2.3"


def test_get_schema_version_from_data_with_uppercase_reference():
    data = {"CTA": {"REFERENCE": {"VERSION": "2.0.0"}}}
    result = schema.get_schema_version_from_data(data)
    assert result == "2.0.0"


def test_get_schema_version_from_data_with_lowercase_reference():
    data = {"cta": {"reference": {"version": "3.1.4"}}}
    result = schema.get_schema_version_from_data(data)
    assert result == "3.1.4"


def test_get_schema_version_from_data_with_no_version():
    data = {"foo": "bar"}
    result = schema.get_schema_version_from_data(data)
    assert result == "latest"


def test_get_schema_version_from_data_with_custom_observatory():
    data = {"VERITAS": {"REFERENCE": {"VERSION": "0.9.8"}}}
    result = schema.get_schema_version_from_data(data, observatory="veritas")
    assert result == "0.9.8"


def test_get_schema_version_from_data_with_custom_observatory_lowercase():
    data = {"veritas": {"reference": {"version": "0.9.9"}}}
    result = schema.get_schema_version_from_data(data, observatory="veritas")
    assert result == "0.9.9"


def test_validate_deprecation_and_version(caplog, monkeypatch):
    """Test validate_deprecation_and_version function covering all edge cases."""

    # Mock simtools version for predictable testing
    def mock_get_software_version(software_name):
        return "1.0.0"

    monkeypatch.setattr(
        "simtools.data_model.schema.get_software_version", mock_get_software_version
    )

    # Test 1: Non-dict data should return early without errors
    schema.validate_deprecation_and_version("not_a_dict")
    schema.validate_deprecation_and_version(None)
    schema.validate_deprecation_and_version([1, 2, 3])

    # Test 2: Empty dict should not raise errors
    schema.validate_deprecation_and_version({})

    # Test 3: Deprecated data should log warning
    with caplog.at_level(logging.WARNING):
        schema.validate_deprecation_and_version({"name": "test_parameter", "deprecated": True})
    assert "Data for test_parameter is deprecated" in caplog.text
    caplog.clear()

    # Test 4: Deprecated data with custom note
    with caplog.at_level(logging.WARNING):
        schema.validate_deprecation_and_version(
            {"deprecated": True, "deprecation_note": "Use new version instead"}
        )
    assert "Use new version instead" in caplog.text
    caplog.clear()

    # Test 5: Non-deprecated data should not warn
    with caplog.at_level(logging.WARNING):
        schema.validate_deprecation_and_version({"deprecated": False})
    assert "deprecated" not in caplog.text

    # Test 6: Valid version constraint should pass
    valid_data = {"simulation_software": [{"name": "simtools", "version": ">=1.0.0"}]}
    schema.validate_deprecation_and_version(valid_data)

    # Test 7: Multiple software entries, only simtools matters
    multi_sw_data = {
        "simulation_software": [
            {"name": "other_software", "version": ">=0.2.0"},
            {"name": "simtools", "version": ">=0.5.0"},
            {"name": "another_software", "version": ">=0.8.0"},
        ]
    }
    schema.validate_deprecation_and_version(multi_sw_data)

    # Test 8: Invalid version constraint should raise ValueError
    invalid_data = {
        "name": "invalid_parameter",
        "simulation_software": [{"name": "simtools", "version": ">=2.0.0"}],
    }
    with pytest.raises(
        ValueError, match=r"invalid_parameter: version 1.0.0 of simtools does not match >=2.0.0"
    ):
        schema.validate_deprecation_and_version(invalid_data)

    # Test 9: Software without version constraint should pass
    no_version_data = {"simulation_software": [{"name": "simtools"}]}
    schema.validate_deprecation_and_version(no_version_data)

    # Test 10: Software with None version should pass
    none_version_data = {"simulation_software": [{"name": "simtools", "version": None}]}
    schema.validate_deprecation_and_version(none_version_data)

    # Test 11: Complex version constraints
    complex_constraints = ["==1.0.0", ">=1.0.0,<2.0.0", "~=1.0", "!=0.9.0"]
    for constraint in complex_constraints:
        data = {"simulation_software": [{"name": "simtools", "version": constraint}]}
        schema.validate_deprecation_and_version(data)

    # Test 12: Version constraint with whitespace should be handled
    whitespace_data = {"simulation_software": [{"name": "simtools", "version": "  >=1.0.0  "}]}
    schema.validate_deprecation_and_version(whitespace_data)

    # Test 12a: Version constraint with random parameter should be handled
    invalid_data = {"simulation_software": [{"name": "simtools", "version": "  >=1.0.0-abc  "}]}
    with pytest.raises(InvalidSpecifier, match=r"Invalid specifier: '>=1.0.0-abc'"):
        schema.validate_deprecation_and_version(invalid_data)

    # Test 13: Custom software name parameter
    custom_sw_data = {"simulation_software": [{"name": "custom_tool", "version": ">=1.0.0"}]}
    schema.validate_deprecation_and_version(custom_sw_data, software_name="custom_tool")

    # Test 13a: Custom software name parameter with mismatch (should skip)
    mismatch_sw_data = {
        "simulation_software": [
            {"name": "other_tool", "version": ">=2.0.0"},
            {"name": "simtools", "version": ">=0.5.0"},
        ]
    }
    schema.validate_deprecation_and_version(mismatch_sw_data, software_name="simtools")

    # Test 14: No matching software name should pass
    no_match_data = {"simulation_software": [{"name": "other_software", "version": ">=0.2.0"}]}
    schema.validate_deprecation_and_version(no_match_data)

    # Test 15: Combined deprecation and version validation
    combined_data = {
        "deprecated": True,
        "deprecation_note": "Old version",
        "simulation_software": [{"name": "simtools", "version": ">=0.5.0"}],
    }
    with caplog.at_level(logging.WARNING):
        schema.validate_deprecation_and_version(combined_data)
    assert "Old version" in caplog.text

    # Test 16: ignore_software_version=True should log warning and not raise
    mismatch_data = {
        "name": "parameter_warning",
        "simulation_software": [{"name": "simtools", "version": ">=2.0.0"}],
    }
    with caplog.at_level(logging.WARNING):
        schema.validate_deprecation_and_version(mismatch_data, ignore_software_version=True)
    assert "does not match" in caplog.text


def test_extract_schema_url_from_metadata_dict():
    """Test _extract_schema_url_from_metadata_dict function."""
    # Test with cta lowercase (default observatory is "cta")
    metadata = {"cta": {"product": {"data": {"model": {"url": "https://schema.example.com"}}}}}
    result = schema._extract_schema_url_from_metadata_dict(metadata)
    assert result == "https://schema.example.com"

    # Test with CTA uppercase and explicit observatory parameter
    metadata = {"CTA": {"product": {"data": {"model": {"url": "https://schema2.example.com"}}}}}
    result = schema._extract_schema_url_from_metadata_dict(metadata, observatory="CTA")
    assert result == "https://schema2.example.com"

    # Test with custom observatory
    metadata = {
        "veritas": {"product": {"data": {"model": {"url": "https://veritas-schema.example.com"}}}}
    }
    result = schema._extract_schema_url_from_metadata_dict(metadata, observatory="veritas")
    assert result == "https://veritas-schema.example.com"

    # Test with no URL
    metadata = {"cta": {"product": {}}}
    result = schema._extract_schema_url_from_metadata_dict(metadata)
    assert result is None

    # Test with empty metadata
    result = schema._extract_schema_url_from_metadata_dict({})
    assert result is None


def test_get_schema_file_from_file_metadata(tmp_test_directory):
    """Test get_schema_file_from_file_metadata function."""
    # Create a test file with schema URL (lowercase cta)
    test_file = Path(tmp_test_directory) / "test_with_schema.yml"
    metadata = {"cta": {"product": {"data": {"model": {"url": "https://schema.example.com"}}}}}
    with open(test_file, "w", encoding="utf-8") as f:
        yaml.dump(metadata, f)

    result = schema.get_schema_file_from_file_metadata(test_file)
    assert result == "https://schema.example.com"

    ecsv_file = Path(tmp_test_directory) / "test_with_schema.ecsv"
    Table(
        rows=[[1]],
        names=["value"],
        meta={"cta": {"product": {"data": {"model": {"url": "https://ecsv-schema.example.com"}}}}},
    ).write(ecsv_file, format="ascii.ecsv")

    result = schema.get_schema_file_from_file_metadata(ecsv_file)
    assert result == "https://ecsv-schema.example.com"

    # Test with non-existent file
    result = schema.get_schema_file_from_file_metadata("non_existent_file.yml")
    assert result is None


def test_get_schema_file_name(tmp_test_directory):
    """Test _get_schema_file_name function."""
    # Test with schema_file provided
    result = schema._get_schema_file_name(schema_file="my_schema.yml")
    assert result == "my_schema.yml"

    # Test with meta_schema_url in data_dict
    data_dict = {"meta_schema_url": "https://schema.example.com"}
    result = schema._get_schema_file_name(data_dict=data_dict)
    assert result == "https://schema.example.com"

    # Test with schema_url in data_dict (e.g. info.yml files)
    data_dict = {"schema_url": "https://info-schema.example.com"}
    result = schema._get_schema_file_name(data_dict=data_dict)
    assert result == "https://info-schema.example.com"

    # Test that meta_schema_url takes precedence over schema_url
    data_dict = {
        "meta_schema_url": "https://meta-schema.example.com",
        "schema_url": "https://schema.example.com",
    }
    result = schema._get_schema_file_name(data_dict=data_dict)
    assert result == "https://meta-schema.example.com"

    # Test with file_name (lowercase cta)
    test_file = Path(tmp_test_directory) / "test_file.yml"
    metadata = {"cta": {"product": {"data": {"model": {"url": "https://file-schema.example.com"}}}}}
    with open(test_file, "w", encoding="utf-8") as f:
        yaml.dump(metadata, f)

    result = schema._get_schema_file_name(file_name=test_file)
    assert result == "https://file-schema.example.com"

    # Test with no inputs
    result = schema._get_schema_file_name()
    assert result is None


def test_validate_schema_from_files(tmp_test_directory, caplog):
    """Test validate_schema_from_files function."""
    # Create a simple valid data file
    test_file = Path(tmp_test_directory) / "valid_data.yml"
    valid_data = {
        "name": "test_parameter",
        "schema_version": "0.1.0",
        "data": [{"value": 1.0}],
    }
    with open(test_file, "w", encoding="utf-8") as f:
        yaml.dump(valid_data, f)

    # Create a simple schema file
    schema_file = Path(tmp_test_directory) / "simple_schema.yml"
    simple_schema = {
        "schema_version": "0.1.0",
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "schema_version": {"type": "string"},
            "data": {"type": "array"},
        },
        "required": ["name", "data"],
    }
    with open(schema_file, "w", encoding="utf-8") as f:
        yaml.dump(simple_schema, f)

    # Test successful validation
    with caplog.at_level(logging.INFO):
        schema.validate_schema_from_files(
            file_directory=tmp_test_directory,
            file_name="valid_data.yml",
            schema_file=schema_file,
            ignore_software_version=True,
        )
    assert "Successful validation of file" in caplog.text

    # Test validation failure
    invalid_file = Path(tmp_test_directory) / "invalid_data.yml"
    invalid_data = {"name": "test_parameter"}
    with open(invalid_file, "w", encoding="utf-8") as f:
        yaml.dump(invalid_data, f)

    with pytest.raises(ValueError, match=r"Validation of file .* failed"):
        schema.validate_schema_from_files(
            file_directory=tmp_test_directory,
            file_name="invalid_data.yml",
            schema_file=schema_file,
            ignore_software_version=True,
        )

    # Test with missing file
    with pytest.raises(FileNotFoundError, match=r"Error reading schema file"):
        schema.validate_schema_from_files(
            file_directory=None,
            file_name="non_existent_file.yml",
            schema_file=schema_file,
        )


def test_table_column_unit_is_optional_in_data_description_schema():
    data_description = {
        "title": "Schema without table column units",
        "schema_version": "0.1.0",
        "meta_schema": "simpipe-schema",
        "meta_schema_version": "0.1.0",
        "name": "schema_without_table_column_units",
        "description": "Table columns without explicit units default to dimensionless.",
        "data": [
            {
                "type": "data_table",
                "table_columns": [
                    {
                        "name": "value",
                        "description": "Dimensionless value.",
                        "required": True,
                        "type": "float64",
                    }
                ],
            }
        ],
    }

    schema.validate_dict_using_schema(
        data_description,
        schema_file=MODEL_PARAMETER_DESCRIPTION_METASCHEMA,
        offline=True,
    )


def test_validate_meta_schema_url_offline():
    """Test _validate_meta_schema_url function."""
    # Test with non-dict data
    schema._validate_meta_schema_url("not a dict")
    schema._validate_meta_schema_url([1, 2, 3])

    # Test with dict without meta_schema_url
    schema._validate_meta_schema_url({"name": "test"})

    # Test with empty meta_schema_url
    with pytest.raises(ValueError, match=r"unknown url type: ''"):
        schema._validate_meta_schema_url({"meta_schema_url": ""})


def test_get_array_element_list(monkeypatch):
    """Test _get_array_element_list function."""

    def mock_array_elements():
        return {"telescope": None, "calibration_device": None}

    def mock_array_element_design_types(element):
        if element == "telescope":
            return ["design1", "design2"]
        return []

    monkeypatch.setattr(schema.names, "array_elements", mock_array_elements)
    monkeypatch.setattr(schema.names, "array_element_design_types", mock_array_element_design_types)

    result = schema._get_array_element_list()
    assert isinstance(result, list)
    assert "telescope" in result
    assert "calibration_device" in result
    assert "telescope-design1" in result
    assert "telescope-design2" in result
