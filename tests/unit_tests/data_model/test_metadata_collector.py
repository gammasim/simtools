#!/usr/bin/python3

import copy
import logging
from pathlib import Path

import pytest
import yaml

import simtools.data_model.metadata_collector as metadata_collector
import simtools.utils.general as gen
from simtools.data_model import metadata_model

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_fill_association_meta_from_args(args_dict_site):
    metadata_1 = metadata_collector.MetadataCollector(args_dict=args_dict_site)
    metadata_1.top_level_meta = gen.change_dict_keys_case(
        metadata_model.top_level_reference_schema(), True
    )
    metadata_1._fill_association_meta_from_args(
        metadata_1.top_level_meta["cta"]["context"]["associated_elements"]
    )

    assert metadata_1.top_level_meta["cta"]["context"]["associated_elements"][0]["site"] == "South"
    assert metadata_1.top_level_meta["cta"]["context"]["associated_elements"][0]["class"] == "MST"
    assert (
        metadata_1.top_level_meta["cta"]["context"]["associated_elements"][0]["type"] == "NectarCam"
    )
    assert metadata_1.top_level_meta["cta"]["context"]["associated_elements"][0]["subtype"] == "D"

    metadata_1.args_dict = None
    with pytest.raises(TypeError):
        metadata_1._fill_association_meta_from_args(
            metadata_1.top_level_meta["cta"]["context"]["associated_elements"]
        )


def test_fill_top_level_meta_from_file(args_dict_site):
    metadata_1 = metadata_collector.MetadataCollector(args_dict=args_dict_site)
    metadata_1.top_level_meta = gen.change_dict_keys_case(
        metadata_model.top_level_reference_schema(), True
    )

    metadata_1.args_dict["input_meta"] = None
    metadata_1._fill_top_level_meta_from_file(metadata_1.top_level_meta["cta"])

    metadata_1.args_dict["input_meta"] = "tests/resources/MLTdata-preproduction.meta.yml"
    metadata_1._fill_top_level_meta_from_file(metadata_1.top_level_meta["cta"])

    assert metadata_1.top_level_meta["cta"]["activity"]["name"] == "mirror_2f_measurement"
    assert metadata_1.top_level_meta["cta"]["context"]["document"][1]["type"] == "Presentation"


def test_fill_product_meta(args_dict_site):
    metadata_1 = metadata_collector.MetadataCollector(args_dict=args_dict_site)
    metadata_1.top_level_meta = gen.change_dict_keys_case(
        metadata_model.top_level_reference_schema(), True
    )

    with pytest.raises(TypeError):
        metadata_1._fill_product_meta(product_dict=None)

    _product_dict = {}
    with pytest.raises(KeyError):
        metadata_1._fill_product_meta(product_dict=_product_dict)

    _product_dict["data"] = {}
    _product_dict["data"]["model"] = {}
    metadata_1._fill_product_meta(product_dict=metadata_1.top_level_meta["cta"]["product"])

    assert metadata_1.top_level_meta["cta"]["product"]["id"] == "UNDEFINED_ACTIVITY_ID"

    assert metadata_1.top_level_meta["cta"]["product"]["data"]["model"]["version"] == "0.0.0"

    # read product metadata from schema file
    metadata_1.args_dict["schema"] = "tests/resources/MST_mirror_2f_measurements.schema.yml"
    metadata_1._fill_product_meta(product_dict=metadata_1.top_level_meta["cta"]["product"])

    assert metadata_1.top_level_meta["cta"]["product"]["data"]["model"]["version"] == "0.1.0"


def test_fill_association_id(args_dict_site):
    metadata_1 = metadata_collector.MetadataCollector(args_dict=args_dict_site)
    metadata_1.top_level_meta = gen.change_dict_keys_case(
        metadata_model.top_level_reference_schema(), True
    )
    metadata_1.top_level_meta["cta"]["context"]["associated_elements"] = get_generic_input_meta()[
        "context"
    ]["associated_elements"]

    metadata_1._fill_association_id(
        metadata_1.top_level_meta["cta"]["context"]["associated_elements"]
    )

    assert (
        metadata_1.top_level_meta["cta"]["context"]["associated_elements"][0]["id"]
        == "South-MST-FlashCam-D"
    )
    assert (
        metadata_1.top_level_meta["cta"]["context"]["associated_elements"][1]["id"]
        == "North-MST-NectarCam-7"
    )

    metadata_1._fill_association_id(
        metadata_1.top_level_meta["cta"]["context"]["associated_elements"]
    )


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
    file_writer_1.top_level_meta = gen.change_dict_keys_case(
        metadata_model.top_level_reference_schema(), True
    )
    file_writer_1._fill_activity_meta(file_writer_1.top_level_meta["cta"]["activity"])

    file_writer_2 = metadata_collector.MetadataCollector(args_dict=args_dict_site)
    file_writer_2.top_level_meta = gen.change_dict_keys_case(
        metadata_model.top_level_reference_schema(), True
    )


def test_fill_context_sim_list(args_dict_site):
    _test_dict_1 = copy.copy(get_generic_input_meta()["context"]["associated_elements"])

    # empty dict -> return same dict
    metadata_collector.MetadataCollector._fill_context_sim_list(_test_dict_1, {})
    assert _test_dict_1 == get_generic_input_meta()["context"]["associated_elements"]

    # add one new entry
    _new_element = {"site": "South", "class": "SCT", "type": "sctcam", "subtype": "7", "id:": None}
    metadata_collector.MetadataCollector._fill_context_sim_list(_test_dict_1, _new_element)
    _test_dict_2 = copy.copy(get_generic_input_meta()["context"]["associated_elements"])
    _test_dict_2.append(_new_element)
    assert _test_dict_1 == _test_dict_2

    # add one new entry to non-existing list -> add
    _test_none = None
    _test_none = metadata_collector.MetadataCollector._fill_context_sim_list(
        _test_none, _new_element
    )
    assert _test_none == [_new_element]
    _test_none = []
    _test_none = metadata_collector.MetadataCollector._fill_context_sim_list(
        _test_none, _new_element
    )
    assert _test_none == [_new_element]

    # one entry with Nones only
    _test_def = [{"site": None, "class": None, "type": None, "subtype": None, "id:": None}]
    _test_def = metadata_collector.MetadataCollector._fill_context_sim_list(_test_def, _new_element)
    assert _test_none == [_new_element]


def test_input_data_file_name(args_dict_site):
    metadata_1 = metadata_collector.MetadataCollector(args_dict=args_dict_site)

    with pytest.raises(KeyError):
        metadata_1.input_data_file_name()

    metadata_1.args_dict["input_data"] = "test.hdf5"
    assert metadata_1.input_data_file_name() == "test.hdf5"


def test_collect_schema_dict(args_dict_site, tmp_test_directory):
    _tmp_schema = get_example_input_schema_single_parameter()
    # write _tmp_schema to a yml file in tmp_test_directory
    _tmp_schema_file = Path(tmp_test_directory).joinpath("ref_long.schema.yml")
    with open(_tmp_schema_file, "w") as outfile:
        yaml.dump(_tmp_schema, outfile, default_flow_style=False)

    metadata_1 = metadata_collector.MetadataCollector(args_dict=args_dict_site)

    assert metadata_1._collect_schema_dict() == {}

    # test when full schema file name is given
    metadata_1.args_dict["schema"] = _tmp_schema_file
    assert metadata_1._collect_schema_dict() == _tmp_schema

    # test when directory including schema file is given, but no parameter name
    metadata_1.args_dict["schema"] = tmp_test_directory
    assert metadata_1._collect_schema_dict() == {}

    # test when directory including schema file is given, and parameter name
    _tmp_parameter = {"name": "ref_long", "value": 1.0}
    _tmp_parameter_file = Path(tmp_test_directory).joinpath("ref_long.yml")
    with open(_tmp_parameter_file, "w") as outfile:
        yaml.dump(_tmp_parameter, outfile, default_flow_style=False)
    metadata_1.args_dict["input"] = _tmp_parameter_file
    # compared sorted dicts, because the order of the keys is not guaranteed
    # (mostly due to above yaml.dump)
    assert dict(sorted(metadata_1._collect_schema_dict().items())) == dict(
        sorted(_tmp_schema.items())
    )


def test_process_metadata_from_file():
    _collector = metadata_collector.MetadataCollector({})

    meta_dict_1 = {"cta": {"product": {"description": "This is a sample description"}}}
    _dict_test_1 = _collector._process_metadata_from_file(meta_dict_1)
    assert _dict_test_1 == meta_dict_1

    meta_dict_2 = {"cta": {"PRODUCT": {"description": "This is a sample description"}}}
    _dict_test_1 = _collector._process_metadata_from_file(meta_dict_2)

    meta_dict_3 = {"cta": {"PRODUCT": {"description": "This is a sample\n description"}}}
    _dict_test_1 = _collector._process_metadata_from_file(meta_dict_3)


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


def test_copy_metadata_from_file(args_dict_site):
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
    _collector._copy_metadata_context_lists(top_level_dict, _input_meta, key)

    assert _result_meta["context"][key] == top_level_dict["context"][key]

    key = "documents"
    with pytest.raises(KeyError):
        _collector._copy_metadata_context_lists(top_level_dict, _input_meta, key)


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
