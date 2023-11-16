import os

import jsonschema
import pytest
import yaml

from simtools.data_model import metadata_model


def test_top_level_reference_schema():
    _top_meta = metadata_model.top_level_reference_schema()

    assert isinstance(_top_meta, dict)
    assert len(_top_meta) > 0

    assert "VERSION" in _top_meta["CTA"]["REFERENCE"]
    assert _top_meta["CTA"]["REFERENCE"]["VERSION"] == "1.0.0"
    assert _top_meta["CTA"]["CONTACT"]["ORGANIZATION"] == "CTAO"


def test_load_schema():
    _metadata_schema = metadata_model.load_schema()
    assert isinstance(_metadata_schema, dict)
    assert len(_metadata_schema) > 0

    with pytest.raises(FileNotFoundError):
        metadata_model.load_schema(schema_file="not_existing_file")


def test_validate_schema(tmp_test_directory):
    sample_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
        "required": ["name", "age"],
    }

    schema_file = os.path.join(tmp_test_directory, "schema.json")
    with open(schema_file, "w", encoding="utf-8") as f:
        yaml.dump(sample_schema, f)

    # sample data dictionary to be validated
    data = {"name": "John", "age": 30}

    metadata_model.validate_schema(data, schema_file)

    with pytest.raises(jsonschema.exceptions.ValidationError):
        invalid_data = {"name": "Alice", "age": "Thirty"}
        metadata_model.validate_schema(invalid_data, schema_file)


def test_resolve_references():
    yaml_data = {
        "example_data": {
            "example_object": {"type": "object", "properties": {"INSTRUMENT": {"type": "string"}}},
            "another_object": {
                "type": "object",
                "properties": {
                    "INSTRUMENT": {"$ref": "#/example_data/example_object/properties/INSTRUMENT"}
                },
            },
        }
    }

    expected_result = {
        "example_data": {
            "example_object": {"type": "object", "properties": {"INSTRUMENT": {"type": "string"}}},
            "another_object": {"type": "object", "properties": {"INSTRUMENT": {"type": "string"}}},
        }
    }

    assert metadata_model.resolve_references(yaml_data) == expected_result


def test_fill_defaults():
    schema = {
        "properties": {
            "CONTACT": {
                "type": "object",
                "properties": {
                    "organization": {"type": "string", "default": "CTA"},
                    "number": {"type": "integer", "default": 30},
                },
            },
            "DOCUMENTS": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "default": "a_document"},
                        "id": {"type": "integer", "default": 55},
                    },
                },
            },
            "NO_DEFAULT": {
                "type": "object",
                "properties": {
                    "string_without_default": {
                        "type": "string",
                    },
                },
            },
            "NO_DEFAULT_LIST": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "string_without_default": {
                            "type": "string",
                        },
                    },
                },
            },
        }
    }
    expected_result = {
        "CTA": {
            "CONTACT": {"organization": "CTA", "number": 30},
            "DOCUMENTS": [{"name": "a_document", "id": 55}],
            "NO_DEFAULT": {},
            "NO_DEFAULT_LIST": [{}],
        }
    }

    assert metadata_model.fill_defaults(schema) == expected_result
