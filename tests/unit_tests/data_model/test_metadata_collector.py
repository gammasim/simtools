#!/usr/bin/python3

import copy
import logging

import pytest

import simtools.data_model.workflow_description as workflow
import simtools.util.general as gen
from simtools.data_model import metadata_model

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_fill_association_id(args_dict_site):

    workflow_1 = workflow.WorkflowDescription(args_dict=args_dict_site)
    workflow_1.top_level_meta = gen.change_dict_keys_case(
        metadata_model.top_level_reference_schema(), True
    )
    workflow_1.top_level_meta["cta"]["context"]["sim"]["association"] = get_generic_input_meta()[
        "product"
    ]["association"]

    workflow_1._fill_association_id(
        workflow_1.top_level_meta["cta"]["context"]["sim"]["association"]
    )

    assert (
        workflow_1.top_level_meta["cta"]["context"]["sim"]["association"][0]["id"]
        == "South-MST-FlashCam-D"
    )
    assert (
        workflow_1.top_level_meta["cta"]["context"]["sim"]["association"][1]["id"]
        == "North-MST-NectarCam-7"
    )

    workflow_1.top_level_meta["cta"]["context"]["sim"]["association"][1]["site"] = "West"
    workflow_1._fill_association_id(
        workflow_1.top_level_meta["cta"]["context"]["sim"]["association"]
    )

    # TODO
    # with pytest.raises(ValueError):
    #    workflow_1._fill_association_id(
    #        workflow_1.top_level_meta["cta"]["context"]["sim"]["association"])


def test_product_data_file_format(args_dict_site):

    workflow_1 = workflow.WorkflowDescription(args_dict=args_dict_site)

    assert workflow_1.product_data_file_format(False) == "ascii.ecsv"
    assert workflow_1.product_data_file_format(True) == "ecsv"

    workflow_1.workflow_config["product"]["format"] = "hdf5"

    assert workflow_1.product_data_file_format(False) == "hdf5"
    assert workflow_1.product_data_file_format(True) == "hdf5"


def test_merge_config_dicts(args_dict_site):

    d_low_priority = {
        "reference": {"version": "0.1.0"},
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
    _workflow = workflow.WorkflowDescription(args_dict=args_dict_site)
    _workflow._merge_config_dicts(d_high_priority, d_low_priority, add_new_fields=True)

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
    _workflow._merge_config_dicts(d_high_priority_2, d_low_priority, add_new_fields=False)

    assert d_merged_no_adding == d_high_priority_2


def test_fill_activity_meta(args_dict_site):

    file_writer_1 = workflow.WorkflowDescription(args_dict=args_dict_site)
    file_writer_1.top_level_meta = gen.change_dict_keys_case(
        metadata_model.top_level_reference_schema(), True
    )
    file_writer_1._fill_activity_meta(file_writer_1.top_level_meta["cta"]["activity"])

    file_writer_2 = workflow.WorkflowDescription(args_dict=args_dict_site)
    file_writer_2.top_level_meta = gen.change_dict_keys_case(
        metadata_model.top_level_reference_schema(), True
    )

    del file_writer_2.workflow_config["activity"]["name"]
    file_writer_2.workflow_config["activity"]["nonmae"] = "workflow_name"

    with pytest.raises(KeyError):
        file_writer_2._fill_activity_meta(file_writer_2.top_level_meta["cta"]["activity"])


def test_fill_context_sim_list(args_dict_site):

    _test_dict_1 = copy.copy(get_generic_input_meta()["product"]["association"])

    # empty dict -> return same dict
    workflow.WorkflowDescription._fill_context_sim_list(_test_dict_1, {})
    assert _test_dict_1 == get_generic_input_meta()["product"]["association"]

    # add one new entry
    _new_element = {"site": "South", "class": "SCT", "type": "sctcam", "subtype": "7", "id:": None}
    workflow.WorkflowDescription._fill_context_sim_list(_test_dict_1, _new_element)
    _test_dict_2 = copy.copy(get_generic_input_meta()["product"]["association"])
    _test_dict_2.append(_new_element)
    assert _test_dict_1 == _test_dict_2

    # add one new entry to non-existing list -> add
    _test_none = None
    _test_none = workflow.WorkflowDescription._fill_context_sim_list(_test_none, _new_element)
    assert _test_none == [_new_element]
    _test_none = []
    _test_none = workflow.WorkflowDescription._fill_context_sim_list(_test_none, _new_element)
    assert _test_none == [_new_element]

    # one entry with Nones only
    _test_def = [{"site": None, "class": None, "type": None, "subtype": None, "id:": None}]
    _test_def = workflow.WorkflowDescription._fill_context_sim_list(_test_def, _new_element)
    assert _test_none == [_new_element]


def get_generic_input_meta():

    return {
        "contact": "my_name",
        "instrument": "my_instrument",
        "product": {
            "description": "my_product",
            "create_time": "2050-01-01",
            "association": [
                {"site": "South", "class": "MST", "type": "FlashCam", "subtype": "D", "id:": None},
                {"site": "North", "class": "MST", "type": "NectarCam", "subtype": "7", "id:": None},
            ],
        },
        "process": "process_description",
    }
