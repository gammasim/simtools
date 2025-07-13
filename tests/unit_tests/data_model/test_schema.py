#!/usr/bin/python3

import logging
from pathlib import Path

import jsonschema
import pytest
import yaml

from simtools.constants import (
    MODEL_PARAMETER_DESCRIPTION_METASCHEMA,
    MODEL_PARAMETER_METASCHEMA,
    MODEL_PARAMETER_SCHEMA_PATH,
)
from simtools.data_model import schema
from simtools.utils import general as gen

DUMMY_FILE = "dummy_file.yml"


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

    schema.validate_dict_using_schema(data, schema_file)

    invalid_data = {"name": "Alice", "age": "Thirty"}
    with pytest.raises(jsonschema.exceptions.ValidationError):
        schema.validate_dict_using_schema(invalid_data, schema_file)

    # with valid meta_schema_url
    data["meta_schema_url"] = "https://github.com/gammasim/simtools"
    schema.validate_dict_using_schema(data, schema_file)

    data["meta_schema_url"] = "https://invalid_url"
    with pytest.raises(FileNotFoundError, match=r"^Meta schema URL does not exist:"):
        schema.validate_dict_using_schema(data, schema_file)


@pytest.mark.xfail(reason="No network connection")
def test_validate_schema_astropy_units(caplog):
    success_string = "Successful validation of data using schema"

    _dict_1 = gen.collect_data_from_file(file_name="tests/resources/num_gains.schema.yml")
    with caplog.at_level(logging.DEBUG):
        schema.validate_dict_using_schema(
            data=_dict_1, schema_file=MODEL_PARAMETER_DESCRIPTION_METASCHEMA
        )
    assert success_string in caplog.text

    # m and cm
    _dict_1["data"][0]["unit"] = "m"
    with caplog.at_level(logging.DEBUG):
        schema.validate_dict_using_schema(
            data=_dict_1, schema_file=MODEL_PARAMETER_DESCRIPTION_METASCHEMA
        )
    assert success_string in caplog.text
    _dict_1["data"][0]["unit"] = "cm"
    with caplog.at_level(logging.DEBUG):
        schema.validate_dict_using_schema(
            data=_dict_1, schema_file=MODEL_PARAMETER_DESCRIPTION_METASCHEMA
        )
    assert success_string in caplog.text

    # combined units
    _dict_1["data"][0]["unit"] = "cm/s"
    with caplog.at_level(logging.DEBUG):
        schema.validate_dict_using_schema(
            data=_dict_1, schema_file=MODEL_PARAMETER_DESCRIPTION_METASCHEMA
        )
    assert success_string in caplog.text
    _dict_1["data"][0]["unit"] = "km/ s"
    with caplog.at_level(logging.DEBUG):
        schema.validate_dict_using_schema(
            data=_dict_1, schema_file=MODEL_PARAMETER_DESCRIPTION_METASCHEMA
        )
    assert success_string in caplog.text

    # dimensionless
    _dict_1["data"][0]["unit"] = "dimensionless"
    with caplog.at_level(logging.DEBUG):
        schema.validate_dict_using_schema(
            data=_dict_1, schema_file=MODEL_PARAMETER_DESCRIPTION_METASCHEMA
        )
    assert success_string in caplog.text
    _dict_1["data"][0]["unit"] = ""
    with caplog.at_level(logging.DEBUG):
        schema.validate_dict_using_schema(
            data=_dict_1, schema_file=MODEL_PARAMETER_DESCRIPTION_METASCHEMA
        )
    assert success_string in caplog.text

    # not good
    _dict_1["data"][0]["unit"] = "not_a_unit"
    with pytest.raises(ValueError, match="'not_a_unit' is not a valid Unit"):
        schema.validate_dict_using_schema(
            data=_dict_1, schema_file=MODEL_PARAMETER_DESCRIPTION_METASCHEMA
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


def test_get_schema_for_version_with_dict():
    test_schema = {"schema_version": "1.0.0", "name": "test"}
    result = schema._get_schema_for_version(test_schema, DUMMY_FILE, "1.0.0")
    assert result == test_schema


def test_get_schema_for_version_with_list_and_latest():
    schema_list = [
        {"schema_version": "2.0.0", "name": "latest"},
        {"schema_version": "1.0.0", "name": "old"},
    ]
    result = schema._get_schema_for_version(schema_list, DUMMY_FILE, "latest")
    assert result["schema_version"] == "2.0.0"


def test_get_schema_for_version_with_list_and_specific_version():
    schema_list = [
        {"schema_version": "2.0.0", "name": "latest"},
        {"schema_version": "1.0.0", "name": "old"},
    ]
    result = schema._get_schema_for_version(schema_list, DUMMY_FILE, "1.0.0")
    assert result["schema_version"] == "1.0.0"


def test_get_schema_for_version_with_list_and_missing_version():
    schema_list = [
        {"schema_version": "2.0.0", "name": "latest"},
        {"schema_version": "1.0.0", "name": "old"},
    ]
    with pytest.raises(ValueError, match="Schema version 3.0.0 not found in dummy_file.yml."):
        schema._get_schema_for_version(schema_list, DUMMY_FILE, "3.0.0")


def test_get_schema_for_version_with_none_version():
    test_schema = {"schema_version": "1.0.0", "name": "test"}
    with pytest.raises(ValueError, match="Schema version not given in dummy_file.yml."):
        schema._get_schema_for_version(test_schema, DUMMY_FILE, None)


def test_get_schema_for_version_with_empty_list():
    schema_list = []
    with pytest.raises(ValueError, match="No schemas found in dummy_file.yml."):
        schema._get_schema_for_version(schema_list, DUMMY_FILE, "latest")


def test_get_schema_for_version_warns_on_version_mismatch(caplog):
    test_schema = {"schema_version": "1.0.0", "name": "test"}
    with caplog.at_level("WARNING"):
        result = schema._get_schema_for_version(test_schema, DUMMY_FILE, "2.0.0")
    assert result == test_schema
    assert "Schema version 2.0.0 does not match 1.0.0" in caplog.text
