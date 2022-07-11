#!/usr/bin/python3

import logging
import re
import pytest

import simtools.util.model_data_writer as writer
import simtools.util.workflow_description as workflow

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_write_metadata(cfg_setup):

    # test writer with no workflow configuration
    w_1 = writer.ModelDataWriter()
    with pytest.raises(
            AttributeError,
            match=r"\'NoneType\' object has no attribute \'product_data_file_name\'"):
        w_1.write_metadata()

    # test writer with workflow configuration initialised, but not set
    c_2 = workflow.WorkflowDescription()
    w_2 = writer.ModelDataWriter(c_2)

    with pytest.raises(
            TypeError,
            match=re.escape('can only concatenate str (not "NoneType") to str')):
        w_2.write_metadata()

    # TODO tests which actually write data

def test_write_data(cfg_setup):

    # test writer with no workflow configuration
    w_1 = writer.ModelDataWriter()
    with pytest.raises(
            AttributeError,
            match=r"\'NoneType\' object has no attribute \'product_data_file_name\'"):
        w_1.write_data(None)

    # test writer with workflow configuration initialised, but not set
    c_2 = workflow.WorkflowDescription()
    w_2 = writer.ModelDataWriter(c_2)
    with pytest.raises(
            TypeError,
            match=re.escape('can only concatenate str (not "NoneType") to str')):
        w_2.write_data(None)

    # TODO tests which actually write data
