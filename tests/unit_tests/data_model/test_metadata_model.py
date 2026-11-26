import pytest

from simtools.data_model import metadata_model


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

    assert metadata_model._resolve_references(yaml_data) == expected_result


def test_fill_defaults():
    schema = {
        "ctao": {
            "properties": {
                "CONTACT": {
                    "type": "object",
                    "properties": {
                        "organization": {"type": "string", "default": "CTAO"},
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
    }
    expected_result = {
        "ctao": {
            "CONTACT": {"organization": "CTAO", "number": 30},
            "DOCUMENTS": [{"name": "a_document", "id": 55}],
            "NO_DEFAULT": {},
            "NO_DEFAULT_LIST": [{}],
        }
    }

    assert metadata_model._fill_defaults(schema) == expected_result

    schema = {
        "ctao": {
            "CONTACT": {
                "type": "object",
                "no_properties": {
                    "organization": {"type": "string", "default": "CTAO"},
                    "number": {"type": "integer", "default": 30},
                },
            },
        }
    }

    with pytest.raises(KeyError, match=r"Missing 'properties' key in schema\."):
        metadata_model._fill_defaults(schema)


def test_add_required_fields():
    metadata = metadata_model.add_required_fields({"ctao": {"product": {"format": "json"}}})

    assert metadata["ctao"]["product"]["format"] == "json"
    assert metadata["ctao"]["product"]["data"]["model"] == {"name": None, "version": None}
    assert metadata["ctao"]["instrument"] == {"class": None, "id": None, "site": None, "type": None}
    assert metadata["ctao"]["contact"] == {"name": None}


def test_get_default_metadata_dict_uses_ctao_block():
    metadata = metadata_model.get_default_metadata_dict()

    assert set(metadata) == {"ctao"}
    assert metadata["ctao"]["reference"]["version"] == "3.0.0"
    assert metadata["ctao"]["contact"]["organization"] == "CTAO"
    # references to '#/definitions/ctao/...' are resolved
    assert "$ref" not in str(metadata)
