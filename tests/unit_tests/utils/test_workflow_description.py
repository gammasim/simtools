#!/usr/bin/python3

import logging

import pytest

import simtools.util.data_model as data_model
import simtools.util.workflow_description as workflow

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_fill_product_association_identifier(args_dict_site):

    workflow_1 = workflow.WorkflowDescription(args_dict=args_dict_site)
    workflow_1.top_level_meta = data_model.top_level_reference_schema()
    workflow_1.top_level_meta["CTA"]["CONTEXT"]["SIM"]["ASSOCIATION"] = get_generic_user_meta()[
        "PRODUCT"
    ]["ASSOCIATION"]
    workflow_1._fill_product_association_identifier()

    workflow_1.top_level_meta["CTA"]["CONTEXT"]["SIM"].pop("ASSOCIATION")

    with pytest.raises(KeyError):
        workflow_1._fill_product_association_identifier()


def test_product_data_file_format(args_dict_site):

    workflow_1 = workflow.WorkflowDescription(args_dict=args_dict_site)

    assert workflow_1.product_data_file_format(False) == "ascii.ecsv"
    assert workflow_1.product_data_file_format(True) == "ecsv"

    workflow_1.workflow_config["PRODUCT"]["FORMAT"] = "hdf5"

    assert workflow_1.product_data_file_format(False) == "hdf5"
    assert workflow_1.product_data_file_format(True) == "hdf5"


def test_merge_config_dicts(args_dict_site):

    d_low_priority = {
        "REFERENCE": {"VERSION": "0.1.0"},
        "ACTIVITY": {"NAME": "SetParameterFromExternal", "DESCRIPTION": "Set data columns"},
        "DATAMODEL": "model-A",
        "PRODUCT": None,
    }

    d_high_priority = {
        "REFERENCE": {"VERSION": "0.2.0"},
        "ACTIVITY": {"NAME": None},
        "PRODUCT": {"DIRECTORY": "./"},
        "DATAMODEL": "model-B",
    }

    _workflow = workflow.WorkflowDescription(args_dict=args_dict_site)
    _workflow._merge_config_dicts(d_high_priority, d_low_priority)

    d_merged = {
        "REFERENCE": {"VERSION": "0.2.0"},
        "ACTIVITY": {"NAME": "SetParameterFromExternal", "DESCRIPTION": "Set data columns"},
        "PRODUCT": {"DIRECTORY": "./"},
        "DATAMODEL": "model-B",
    }

    assert d_merged == d_high_priority


def test_fill_activity_meta(args_dict_site):

    file_writer_1 = workflow.WorkflowDescription(args_dict=args_dict_site)
    file_writer_1.top_level_meta = data_model.top_level_reference_schema()
    file_writer_1._fill_activity_meta()

    file_writer_2 = workflow.WorkflowDescription(args_dict=args_dict_site)
    file_writer_2.top_level_meta = data_model.top_level_reference_schema()

    del file_writer_2.workflow_config["ACTIVITY"]["NAME"]
    file_writer_2.workflow_config["ACTIVITY"]["NONAME"] = "workflow_name"

    with pytest.raises(KeyError):
        file_writer_2._fill_activity_meta()


def get_generic_user_meta():

    return {
        "CONTACT": "my_name",
        "INSTRUMENT": "my_instrument",
        "PRODUCT": {
            "DESCRIPTION": "my_product",
            "CREATION_TIME": "2050-01-01",
            "ASSOCIATION": [
                {"SITE": "South", "CLASS": "MST", "TYPE": "FlashCam", "SUBTYPE": "D", "ID:": None},
                {"SITE": "North", "CLASS": "MST", "TYPE": "NectarCam", "SUBTYPE": "7", "ID:": None},
            ],
        },
        "PROCESS": "process_description",
    }
