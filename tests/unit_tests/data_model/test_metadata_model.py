import logging

import pytest

from simtools.data_model import metadata_model


def test_get_default_metadata_dict():
    _top_meta = metadata_model.get_default_metadata_dict()

    assert isinstance(_top_meta, dict)
    assert len(_top_meta) > 0

    assert "VERSION" in _top_meta["CTA"]["REFERENCE"]
    assert _top_meta["CTA"]["REFERENCE"]["VERSION"] == "1.0.0"
    assert _top_meta["CTA"]["CONTACT"]["ORGANIZATION"] == "CTAO"


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


def test_fill_defaults(caplog):
    schema = {
        "CTA": {
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
    }
    expected_result = {
        "CTA": {
            "CONTACT": {"organization": "CTA", "number": 30},
            "DOCUMENTS": [{"name": "a_document", "id": 55}],
            "NO_DEFAULT": {},
            "NO_DEFAULT_LIST": [{}],
        }
    }

    assert metadata_model._fill_defaults(schema) == expected_result

    schema = {
        "CTA": {
            "CONTACT": {
                "type": "object",
                "no_properties": {
                    "organization": {"type": "string", "default": "CTA"},
                    "number": {"type": "integer", "default": 30},
                },
            },
        }
    }

    with caplog.at_level(logging.ERROR):
        with pytest.raises(KeyError):
            metadata_model._fill_defaults(schema)

    assert "Missing 'properties' key in schema." in caplog.text
