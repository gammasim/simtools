import copy

import pytest

from simtools.data_model import meta_data_model


def test_top_level_reference_schema():

    _top_meta = meta_data_model.top_level_reference_schema()

    assert isinstance(_top_meta, dict)
    assert len(_top_meta) > 0

    assert "VERSION" in _top_meta["CTA"]["REFERENCE"]


def test_metadata_input_reference_schema():
    """(very hard to test this)"""

    _top_ref = meta_data_model.metadata_input_reference_schema()

    assert isinstance(_top_ref, dict)
    assert len(_top_ref) > 0

    assert "VERSION" in _top_ref["REFERENCE"]


def test_workflow_configuration_schema():

    _config = meta_data_model.workflow_configuration_schema()

    assert isinstance(_config, dict)
    assert len(_config) > 0

    assert "configuration" in _config


def test_metadata_input_reference_document_list():

    assert "SITE" in meta_data_model.metadata_input_reference_document_list("instrumentlist")
    assert "SITE" in meta_data_model.metadata_input_reference_document_list("INSTRUMENTLIST")
    assert "TYPE" in meta_data_model.metadata_input_reference_document_list("documentlist")
    with pytest.raises(meta_data_model.InvalidSchemaList, match=r"Invalid schema list: wronglist"):
        meta_data_model.metadata_input_reference_document_list("wronglist")


def test_metadata_dict_with_defaults():

    _test_dict = {
        "INSTRUMENT": {
            "SITE": {"type": "str", "required": True, "default": "North"},
            "SUBTYPE": {"type": "str", "required": False, "default": None},
        }
    }
    assert meta_data_model._metadata_dict_with_defaults(_test_dict) == {
        "INSTRUMENT": {"SITE": "North", "SUBTYPE": None}
    }
    _test_dict_2 = {
        "INSTRUMENT": {
            "SITE": {"type": "str", "required": True},
            "SUBTYPE": {"type": "str", "required": False, "default": None},
        }
    }

    with pytest.raises(
        meta_data_model.InvalidSchemaList,
        match=r"Invalid schema list with missing type, required, or default fields",
    ):
        meta_data_model._metadata_dict_with_defaults(_test_dict_2)


def test_remove_empty_lists():

    _test_dict_1 = {"SITE": {"type": "str", "required": True, "default": "North"}}

    assert meta_data_model._remove_empty_lists(copy.deepcopy(_test_dict_1)) == _test_dict_1

    _test_dict_2 = {
        "SITE": {"type": "str", "required": True, "default": "North"},
        "ASSOCIATION": [],
    }
    assert meta_data_model._remove_empty_lists(copy.deepcopy(_test_dict_2)) == _test_dict_1
