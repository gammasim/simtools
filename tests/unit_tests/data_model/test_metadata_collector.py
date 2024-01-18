#!/usr/bin/python3

import copy
import getpass
import json
import logging

import pytest

import simtools.data_model.metadata_collector as metadata_collector
import simtools.utils.general as gen
from simtools.data_model import metadata_model

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_get_data_model_schema_file_name():
    # from args_dict / command line
    args_dict = {"no_schema": "schema_file.yml"}
    _collector = metadata_collector.MetadataCollector(args_dict)
    schema_file = _collector.get_data_model_schema_file_name()
    assert schema_file is None

    args_dict = {"schema": "simtools/schemas/metadata.metaschema.yml"}
    _collector = metadata_collector.MetadataCollector(args_dict)
    schema_file = _collector.get_data_model_schema_file_name()
    assert schema_file == args_dict["schema"]

    # from metadata
    _collector.top_level_meta["cta"]["product"]["data"]["model"][
        "url"
    ] = "simtools/schemas/top_level_meta.schema.yml"
    schema_file = _collector.get_data_model_schema_file_name()
    # test that priority is given to args_dict (if not none)
    assert schema_file == args_dict["schema"]
    _collector.args_dict["schema"] = None
    schema_file = _collector.get_data_model_schema_file_name()
    assert schema_file == "simtools/schemas/top_level_meta.schema.yml"

    _collector.top_level_meta["cta"]["product"]["data"]["model"].pop("url")
    schema_file = _collector.get_data_model_schema_file_name()
    assert schema_file is None

    # from data model_name
    _collector.data_model_name = "array_coordinates"
    schema_file = _collector.get_data_model_schema_file_name()
    url = "https://raw.githubusercontent.com/gammasim/workflows/main/schemas/"
    url += "array_coordinates.schema.yml"
    assert schema_file == url

    # from input metadata
    _collector.input_metadata = {
        "cta": {"product": {"data": {"model": {"url": "from_input_meta"}}}}
    }
    _collector.data_model_name = None
    schema_file = _collector.get_data_model_schema_file_name()
    assert schema_file == "from_input_meta"


def test_get_data_model_schema_dict(args_dict_site):
    metadata = metadata_collector.MetadataCollector(args_dict=args_dict_site)
    metadata.schema_file_name = "simtools/schemas/metadata.metaschema.yml"

    assert isinstance(metadata.get_data_model_schema_dict(), dict)

    metadata.schema_file_name = "this_file_does_not_exist"
    assert metadata.get_data_model_schema_dict() == {}


def test_fill_contact_meta(args_dict_site):
    contact_dict = {}
    collector = metadata_collector.MetadataCollector(args_dict=args_dict_site)
    collector._fill_contact_meta(contact_dict)
    assert contact_dict["name"] == getpass.getuser()


def test_fill_associated_elements_from_args(args_dict_site):
    metadata_1 = metadata_collector.MetadataCollector(args_dict=args_dict_site)
    metadata_1.top_level_meta = gen.change_dict_keys_case(
        metadata_model.get_default_metadata_dict(), True
    )
    metadata_1._fill_associated_elements_from_args(
        metadata_1.top_level_meta["cta"]["context"]["associated_elements"]
    )

    assert metadata_1.top_level_meta["cta"]["context"]["associated_elements"][0]["site"] == "South"
    assert metadata_1.top_level_meta["cta"]["context"]["associated_elements"][0]["class"] == "MST"
    assert (
        metadata_1.top_level_meta["cta"]["context"]["associated_elements"][0]["type"] == "NectarCam"
    )
    assert metadata_1.top_level_meta["cta"]["context"]["associated_elements"][0]["subtype"] == "D"

    metadata_1.top_level_meta["cta"]["context"]["associated_elements"][0].pop("site")

    metadata_1.args_dict = None
    with pytest.raises(TypeError):
        metadata_1._fill_associated_elements_from_args(
            metadata_1.top_level_meta["cta"]["context"]["associated_elements"]
        )


def test_read_input_metadata_from_file(args_dict_site, tmp_test_directory, caplog):
    metadata_1 = metadata_collector.MetadataCollector(args_dict=args_dict_site)
    metadata_1.args_dict["input_meta"] = None

    assert metadata_1._read_input_metadata_from_file() == {}

    metadata_1.args_dict["input_meta"] = "./file_does_not_exist.yml"
    with pytest.raises(FileNotFoundError):
        metadata_1._read_input_metadata_from_file()

    metadata_1.args_dict["input_meta"] = "./file_does_not_exist.not_a_good_suffix"
    with pytest.raises(gen.InvalidConfigData):
        metadata_1._read_input_metadata_from_file()

    metadata_1.args_dict["input_meta"] = "tests/resources/MLTdata-preproduction.meta.yml"
    assert len(metadata_1._read_input_metadata_from_file()) > 0

    metadata_1.args_dict["input_meta"] = "tests/resources/reference_point_altitude.json"
    assert len(metadata_1._read_input_metadata_from_file()) > 0

    test_dict = {
        "metadata": {"cta": {"product": {"data": {"model": {"url": "from_input_meta"}}}}},
        "METADATA": {"cta": {"product": {"data": {"model": {"url": "from_input_meta"}}}}},
    }
    with open(tmp_test_directory / "test_read_input_metadata_file.json", "w") as f:
        json.dump(test_dict, f)
    metadata_1.args_dict["input_meta"] = tmp_test_directory / "test_read_input_metadata_file.json"
    with pytest.raises(gen.InvalidConfigData):
        metadata_1._read_input_metadata_from_file()
        assert "More than one metadata entry" in caplog.text


def test_fill_context_from_input_meta(args_dict_site):
    metadata_1 = metadata_collector.MetadataCollector(args_dict=args_dict_site)

    metadata_1.args_dict["input_meta"] = "tests/resources/MLTdata-preproduction.meta.yml"
    metadata_1.input_metadata = metadata_1._read_input_metadata_from_file()
    metadata_1._fill_context_from_input_meta(metadata_1.top_level_meta["cta"]["context"])

    assert metadata_1.top_level_meta["cta"]["context"]["document"][1]["type"] == "Presentation"
    assert (
        metadata_1.top_level_meta["cta"]["context"]["associated_data"][0]["description"][0:6]
        == "Mirror"
    )


def test_fill_product_meta(args_dict_site):
    metadata_1 = metadata_collector.MetadataCollector(args_dict=args_dict_site)

    with pytest.raises(TypeError):
        metadata_1._fill_product_meta(product_dict=None)

    _product_dict = {}
    with pytest.raises(KeyError):
        metadata_1._fill_product_meta(product_dict=_product_dict)

    _product_dict["data"] = {}
    _product_dict["data"]["model"] = {}
    metadata_1._fill_product_meta(product_dict=metadata_1.top_level_meta["cta"]["product"])

    assert metadata_1.top_level_meta["cta"]["product"]["id"] == "UNDEFINED_ACTIVITY_ID"

    assert metadata_1.top_level_meta["cta"]["product"]["data"]["model"]["version"] is None

    # read product metadata from schema file
    metadata_1.args_dict["schema"] = "tests/resources/MST_mirror_2f_measurements.schema.yml"
    metadata_1._fill_product_meta(product_dict=metadata_1.top_level_meta["cta"]["product"])

    assert metadata_1.top_level_meta["cta"]["product"]["data"]["model"]["version"] == "0.1.0"


def test_fill_process_meta(args_dict_site):
    metadata_1 = metadata_collector.MetadataCollector(args_dict=args_dict_site)
    metadata_1._fill_process_meta(metadata_1.top_level_meta["cta"]["activity"])

    assert metadata_1.top_level_meta["cta"]["activity"]["type"] == "simulation"


def test_merge_config_dicts(args_dict_site):
    d_low_priority = {
        "reference": {"version": "1.0.0"},
        "activity": {"name": "SetParameterFromExternal", "description": "Set data columns"},
        "datamodel": "model-A",
        "product": None,
    }

    d_high_priority = {
        "reference": {"version": "0.2.0"},
        "activity": {"name": None},
        "product": {"directory": "./"},
        "datamodel": "model-B",
    }
    d_high_priority_2 = copy.deepcopy(d_high_priority)
    # this should do only some thing with d_high_priority!
    _metadata = metadata_collector.MetadataCollector(args_dict=args_dict_site)
    _metadata._merge_config_dicts(d_high_priority, d_low_priority, add_new_fields=True)

    d_merged = {
        "reference": {"version": "0.2.0"},
        "activity": {"name": "SetParameterFromExternal", "description": "Set data columns"},
        "product": {"directory": "./"},
        "datamodel": "model-B",
    }
    assert d_merged == d_high_priority

    d_merged_no_adding = {
        "reference": {"version": "0.2.0"},
        "activity": {"name": "SetParameterFromExternal"},
        "product": {"directory": "./"},
        "datamodel": "model-B",
    }
    _metadata._merge_config_dicts(d_high_priority_2, d_low_priority, add_new_fields=False)

    assert d_merged_no_adding == d_high_priority_2


def test_fill_activity_meta(args_dict_site):
    file_writer_1 = metadata_collector.MetadataCollector(args_dict=args_dict_site)
    file_writer_1._fill_activity_meta(file_writer_1.top_level_meta["cta"]["activity"])

    # this is set by args_dict_site in conf.py  (although this is a unit test)
    assert file_writer_1.top_level_meta["cta"]["activity"]["name"] == "integration_test"


def test_fill_context_sim_list(args_dict_site):
    _test_dict_1 = copy.copy(get_generic_input_meta()["context"]["associated_elements"])

    # empty dict -> return same dict
    _collector = metadata_collector.MetadataCollector(args_dict=args_dict_site)

    _collector._fill_context_sim_list(meta_list=_test_dict_1, new_entry_dict={})
    assert _test_dict_1 == get_generic_input_meta()["context"]["associated_elements"]

    # add one new entry
    _new_element = {"site": "South", "class": "SCT", "type": "sctcam", "subtype": "7", "id:": None}
    _collector._fill_context_sim_list(_test_dict_1, _new_element)
    _test_dict_2 = copy.copy(get_generic_input_meta()["context"]["associated_elements"])
    _test_dict_2.append(_new_element)
    assert _test_dict_1 == _test_dict_2

    # add one new entry to non-existing list -> add
    _test_none = None
    _test_none = _collector._fill_context_sim_list(_test_none, _new_element)
    assert _test_none == [_new_element]
    _test_none = []
    _test_none = _collector._fill_context_sim_list(_test_none, _new_element)
    assert _test_none == [_new_element]

    # one entry with Nones only
    _test_def = [{"site": None, "class": None, "type": None, "subtype": None, "id:": None}]
    _test_def = _collector._fill_context_sim_list(_test_def, _new_element)
    assert _test_none == [_new_element]


def test_process_metadata_from_file():
    _collector = metadata_collector.MetadataCollector({})

    meta_dict_1 = {"cta": {"product": {"description": "This is a sample description"}}}
    _dict_test_1 = _collector._process_metadata_from_file(meta_dict_1)
    assert _dict_test_1 == meta_dict_1

    meta_dict_2 = {"cta": {"PRODUCT": {"description": "This is a sample description"}}}
    _dict_test_1 = _collector._process_metadata_from_file(meta_dict_2)

    meta_dict_3 = {"cta": {"PRODUCT": {"description": "This is a sample\n description"}}}
    _dict_test_1 = _collector._process_metadata_from_file(meta_dict_3)

    meta_dict_4 = {"cta": {"product": {"description": None}}}
    assert _collector._process_metadata_from_file(meta_dict_4) == meta_dict_4


def test__remove_line_feed():
    collector = metadata_collector.MetadataCollector({})
    input_string = "This is a string without line feeds."
    result = collector._remove_line_feed(input_string)
    assert result == input_string

    input_string_2 = "This is a string\n with line feeds."
    result = collector._remove_line_feed(input_string_2)
    assert result == input_string.replace("without", "with")

    input_string_3 = "This is a string\r with line feeds."
    result = collector._remove_line_feed(input_string_3)
    assert result == input_string.replace("without", "with")

    assert "" == collector._remove_line_feed("")

    assert " " == collector._remove_line_feed(" ")

    assert " " == collector._remove_line_feed("  ")


def test_copy_list_type_metadata(args_dict_site):
    top_level_dict = {
        "context": {
            "associated_elements": [
                {"site": "Site A", "class": "Class A", "type": "Type A", "subtype": "Subtype A"}
            ]
        }
    }

    _input_meta = {
        "context": {
            "associated_elements": [
                {"site": "Site B", "class": "Class B", "type": "Type B", "subtype": "Subtype B"},
                {"site": "Site C", "class": "Class C", "type": "Type C", "subtype": "Subtype C"},
            ]
        }
    }

    _result_meta = {
        "context": {
            "associated_elements": [
                {"site": "Site A", "class": "Class A", "type": "Type A", "subtype": "Subtype A"},
                {"site": "Site B", "class": "Class B", "type": "Type B", "subtype": "Subtype B"},
                {"site": "Site C", "class": "Class C", "type": "Type C", "subtype": "Subtype C"},
            ]
        }
    }

    key = "associated_elements"

    _collector = metadata_collector.MetadataCollector({})
    _collector._copy_list_type_metadata(top_level_dict["context"], _input_meta, key)

    assert _result_meta["context"][key] == top_level_dict["context"][key]

    key = "documents"
    _org_top_level_dict = copy.deepcopy(top_level_dict)
    _collector._copy_list_type_metadata(top_level_dict, _input_meta, key)
    assert _org_top_level_dict == top_level_dict


def test_input_dict_is_none():
    _collector = metadata_collector.MetadataCollector(args_dict={})
    input_dict = None
    result = _collector._all_values_none(input_dict)
    assert result is True

    input_dict = "not a dictionary"
    result = _collector._all_values_none(input_dict)
    assert result is False

    input_dict = {"key1": None, "key2": None}
    result = _collector._all_values_none(input_dict)
    assert result is True

    input_dict = {}
    result = _collector._all_values_none(input_dict)
    assert result is True

    input_dict = {"key1": None, "key2": "value"}
    result = _collector._all_values_none(input_dict)
    assert result is False


def get_generic_input_meta():
    return {
        "contact": "my_name",
        "instrument": "my_instrument",
        "product": {
            "description": "my_product",
            "create_time": "2050-01-01",
        },
        "process": "process_description",
        "context": {
            "associated_elements": [
                {"site": "South", "class": "MST", "type": "FlashCam", "subtype": "D", "id:": None},
                {"site": "North", "class": "MST", "type": "NectarCam", "subtype": "7", "id:": None},
            ],
        },
    }


def get_example_input_schema_single_parameter():
    return {
        "version": "1.0.0",
        "name": "ref_lat",
        "description": "Latitude of site centre.",
        "short_description": "Latitude of site centre.",
        "data": [{"type": "double", "units": "deg", "allowed_range": {"min": -90.0, "max": 90.0}}],
    }
