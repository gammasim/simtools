import copy

import pytest

from simtools.data_model import metadata_model


def test_top_level_reference_schema():

    _top_meta = metadata_model.top_level_reference_schema()

    assert isinstance(_top_meta, dict)
    assert len(_top_meta) > 0

    assert "VERSION" in _top_meta["CTA"]["REFERENCE"]


def test_metadata_input_reference_schema():
    """(very hard to test this)"""

    _top_ref = metadata_model.metadata_input_reference_schema()

    assert isinstance(_top_ref, dict)
    assert len(_top_ref) > 0

    assert "VERSION" in _top_ref["REFERENCE"]


def test_metadata_input_reference_document_list():

    assert "SITE" in metadata_model.metadata_input_reference_document_list("instrumentlist")
    assert "SITE" in metadata_model.metadata_input_reference_document_list("INSTRUMENTLIST")
    assert "TYPE" in metadata_model.metadata_input_reference_document_list("documentlist")
    with pytest.raises(metadata_model.InvalidSchemaList, match=r"Invalid schema list: wronglist"):
        metadata_model.metadata_input_reference_document_list("wronglist")


def test_metadata_dict_with_defaults():

    _test_dict = {
        "INSTRUMENT": {
            "SITE": {"type": "str", "required": True, "default": "North"},
            "SUBTYPE": {"type": "str", "required": False, "default": None},
        }
    }
    assert metadata_model._metadata_dict_with_defaults(_test_dict) == {
        "INSTRUMENT": {"SITE": "North", "SUBTYPE": None}
    }
    _test_dict_2 = {
        "INSTRUMENT": {
            "SITE": {"type": "str", "required": True},
            "SUBTYPE": {"type": "str", "required": False, "default": None},
        }
    }

    with pytest.raises(
        metadata_model.InvalidSchemaList,
        match=r"Invalid schema list with missing type, required, or default fields",
    ):
        metadata_model._metadata_dict_with_defaults(_test_dict_2)


def test_remove_empty_lists():

    _test_dict_1 = {"SITE": {"type": "str", "required": True, "default": "North"}}

    assert metadata_model._remove_empty_lists(copy.deepcopy(_test_dict_1)) == _test_dict_1

    _test_dict_2 = {
        "SITE": {"type": "str", "required": True, "default": "North"},
        "ASSOCIATION": [],
    }
    assert metadata_model._remove_empty_lists(copy.deepcopy(_test_dict_2)) == _test_dict_1
