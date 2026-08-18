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

_INTEGRATION_CONFIG_FILES = sorted(
    (Path(__file__).parents[2] / "integration_tests" / "config").glob("*.yml")
)


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


def test_get_model_parameter_schema_returns_independent_copies():
    schema_loader.clear_cache()
    schema_1 = schema.get_model_parameter_schema("mirror_focal_length", "0.1.0")
    schema_2 = schema.get_model_parameter_schema("mirror_focal_length", "0.1.0")

    schema_1["data"][0]["unit"] = "m"

    assert schema_2["data"][0]["unit"] == "cm"


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


def _output_validation_workflow(*validations):
    """Build a minimal workflow containing composable output validators."""
    return {
        "schema_version": "0.5.0",
        "schema_name": "application_workflow.metaschema",
        "applications": [
            {
                "application": "simtools-test",
                "configuration": {"output_path": "output"},
                "integration_tests": [
                    {
                        "test_outputs": [
                            {
                                "path_descriptor": "output_path",
                                "file": "output.ecsv",
                                "validations": list(validations),
                            }
                        ]
                    }
                ],
            }
        ],
    }


def _valid_output_validation_rule():
    """Build a representative table validator for metaschema tests."""
    return {
        "type": "table",
        "minimum_rows": 1,
        "unique_columns": ["id"],
        "columns": {
            "energy": {
                "range": {"minimum": 1.0, "unit": "GeV"},
            },
        },
    }


def test_application_workflow_schema_accepts_output_validators():
    """Test composable output-validator configuration shapes."""
    workflow_config = _output_validation_workflow(
        {"type": "data_schema", "schema": "schema.yml"},
        _valid_output_validation_rule(),
        {
            "type": "metadata",
            "required_keys": ["summary"],
            "relations": [{"left": "summary.rows", "equals": "table.row_count"}],
        },
    )

    schema.validate_dict_using_schema(
        workflow_config,
        schema_file=SCHEMA_PATH / "application_workflow.metaschema.yml",
    )


def test_application_workflow_schema_accepts_resource_benchmark_exclusion():
    """Allow a reasoned opt-out from CI resource benchmarking."""
    workflow_config = _output_validation_workflow()
    workflow_config["applications"][0].pop("integration_tests")
    workflow_config["applications"][0]["exclude_from_resource_benchmark"] = "unstable service"

    schema.validate_dict_using_schema(
        workflow_config,
        schema_file=SCHEMA_PATH / "application_workflow.metaschema.yml",
    )


def test_application_workflow_schema_rejects_empty_resource_benchmark_exclusion():
    """Require a non-empty reason for a resource benchmark opt-out."""
    workflow_config = _output_validation_workflow()
    workflow_config["applications"][0].pop("integration_tests")
    workflow_config["applications"][0]["exclude_from_resource_benchmark"] = ""

    with pytest.raises(jsonschema.ValidationError):
        schema.validate_dict_using_schema(
            workflow_config,
            schema_file=SCHEMA_PATH / "application_workflow.metaschema.yml",
        )


def test_application_workflow_schema_accepts_profiled_output_validation_rule():
    """Allow profiles to provide shared output-validation fields."""
    workflow_config = _output_validation_workflow({"profile": "job_grid", "file": "job_grid.ecsv"})

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
    ],
)
def test_application_workflow_schema_rejects_malformed_output_validator(change):
    """Reject unknown properties and malformed output validators."""
    rule = _valid_output_validation_rule()
    change(rule)
    workflow_config = _output_validation_workflow(rule)

    with pytest.raises(jsonschema.ValidationError):
        schema.validate_dict_using_schema(
            workflow_config,
            schema_file=SCHEMA_PATH / "application_workflow.metaschema.yml",
        )


def test_application_workflow_schema_rejects_legacy_output_fields():
    """Reject the removed output-validation interface."""
    workflow_config = _output_validation_workflow(_valid_output_validation_rule())
    integration_test = workflow_config["applications"][0]["integration_tests"][0]
    integration_test["output_file"] = "legacy.ecsv"

    with pytest.raises(jsonschema.ValidationError):
        schema.validate_dict_using_schema(
            workflow_config,
            schema_file=SCHEMA_PATH / "application_workflow.metaschema.yml",
        )


def test_application_workflow_schema_preserves_previous_version():
    """Load the newest workflow schema first while retaining version 0.4.0."""
    schema_file = SCHEMA_PATH / "application_workflow.metaschema.yml"

    assert schema.load_schema(schema_file)["schema_version"] == "0.5.0"
    assert schema.load_schema(schema_file, "0.4.0")["schema_version"] == "0.4.0"

    legacy_workflow = {
        "schema_version": "0.4.0",
        "schema_name": "application_workflow.metaschema",
        "applications": [
            {
                "application": "simtools-test",
                "configuration": {"output_path": "output"},
                "integration_tests": [{"output_file": "legacy.ecsv"}],
            }
        ],
    }
    schema.validate_dict_using_schema(legacy_workflow, schema_file=schema_file)


def test_application_workflow_schema_accepts_typed_reference_options():
    """Accept explicit filtering and key-based ECSV reference comparison."""
    workflow_config = _output_validation_workflow(
        {
            "type": "reference",
            "file": "reference.ecsv",
            "key_columns": ["id"],
            "filters": [{"column": "primary", "operator": "equal", "value": "gamma"}],
        }
    )

    schema.validate_dict_using_schema(
        workflow_config,
        schema_file=SCHEMA_PATH / "application_workflow.metaschema.yml",
    )


@pytest.mark.parametrize("config_file", _INTEGRATION_CONFIG_FILES, ids=lambda path: path.stem)
def test_integration_configs_match_application_workflow_schema(config_file):
    """Validate every maintained integration configuration against its workflow schema."""
    schema.validate_dict_using_schema(
        ascii_handler.collect_data_from_file(config_file),
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
    data = {
        "schema_version": "0.2.0",
        "model_version": "6.1.0",
        "model_update": "patch_update",
        "model_version_history": ["6.0.2"],
        "model_status": model_status,
        "description": "test",
        "changes": {},
    }

    assert (
        schema.validate_dict_using_schema(
            data=data,
            schema_file="simulation_models_info.schema.yml",
            offline=True,
        )
        == data
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


def _mock_software_version(monkeypatch):
    monkeypatch.setattr("simtools.data_model.schema.get_software_version", lambda _: "1.0.0")


@pytest.mark.parametrize(
    "data",
    ["not_a_dict", None, [1, 2, 3], {}, {"deprecated": False}],
    ids=["non-mapping", "none", "list", "empty", "not-deprecated"],
)
def test_validate_deprecation_and_version_accepts_noop_inputs(data, caplog, monkeypatch):
    _mock_software_version(monkeypatch)
    with caplog.at_level(logging.WARNING):
        assert schema.validate_deprecation_and_version(data) is None
    assert caplog.text == ""


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"name": "test_parameter", "deprecated": True}, "Data for test_parameter is deprecated"),
        (
            {"deprecated": True, "deprecation_note": "Use new version instead"},
            "Use new version instead",
        ),
    ],
    ids=["default-note", "custom-note"],
)
def test_validate_deprecation_and_version_warns_for_deprecated_data(
    data, message, caplog, monkeypatch
):
    _mock_software_version(monkeypatch)
    with caplog.at_level(logging.WARNING):
        assert schema.validate_deprecation_and_version(data) is None
    assert message in caplog.text


@pytest.mark.parametrize(
    "constraint",
    ["==1.0.0", ">=1.0.0,<2.0.0", "~=1.0", "!=0.9.0", "  >=1.0.0  "],
    ids=["exact", "range", "compatible", "not-equal", "whitespace"],
)
def test_validate_deprecation_and_version_accepts_valid_constraints(constraint, monkeypatch):
    _mock_software_version(monkeypatch)
    data = {"simulation_software": [{"name": "simtools", "version": constraint}]}
    assert schema.validate_deprecation_and_version(data) is None


@pytest.mark.parametrize(
    ("data", "software_name"),
    [
        ({"simulation_software": [{"name": "simtools"}]}, "simtools"),
        ({"simulation_software": [{"name": "simtools", "version": None}]}, "simtools"),
        ({"simulation_software": [{"name": "custom_tool", "version": ">=1.0.0"}]}, "custom_tool"),
        ({"simulation_software": [{"name": "other_software", "version": ">=0.2.0"}]}, "simtools"),
    ],
    ids=["missing-version", "none-version", "custom-name", "no-match"],
)
def test_validate_deprecation_and_version_accepts_optional_or_unrelated_constraints(
    data, software_name, monkeypatch
):
    _mock_software_version(monkeypatch)
    assert schema.validate_deprecation_and_version(data, software_name=software_name) is None


def test_validate_deprecation_and_version_reports_invalid_constraints(caplog, monkeypatch):
    _mock_software_version(monkeypatch)
    mismatch = {
        "name": "invalid_parameter",
        "simulation_software": [{"name": "simtools", "version": ">=2.0.0"}],
    }
    with pytest.raises(ValueError, match=r"invalid_parameter: version 1\.0\.0"):
        schema.validate_deprecation_and_version(mismatch)

    with pytest.raises(InvalidSpecifier, match=r"Invalid specifier: '>=1.0.0-abc'"):
        schema.validate_deprecation_and_version(
            {"simulation_software": [{"name": "simtools", "version": ">=1.0.0-abc"}]}
        )

    with caplog.at_level(logging.WARNING):
        assert (
            schema.validate_deprecation_and_version(mismatch, ignore_software_version=True) is None
        )
    assert "does not match" in caplog.text


def test_extract_schema_url_from_metadata_dict():
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


def test_validate_meta_schema_url_offline():
    # Test with non-dict data
    schema._validate_meta_schema_url("not a dict")
    schema._validate_meta_schema_url([1, 2, 3])

    # Test with dict without meta_schema_url
    schema._validate_meta_schema_url({"name": "test"})

    # Test with empty meta_schema_url
    with pytest.raises(ValueError, match=r"unknown url type: ''"):
        schema._validate_meta_schema_url({"meta_schema_url": ""})
