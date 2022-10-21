#!/usr/bin/python3

import logging
import re

import pytest

import simtools.util.model_data_writer as writer
import simtools.util.workflow_description as workflow

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_write_metadata(args_dict_site, io_handler):

    # test writer with no workflow configuration
    w_1 = writer.ModelDataWriter()
    with pytest.raises(
        AttributeError, match=r"\'NoneType\' object has no attribute \'product_data_file_name\'"
    ):
        w_1.write_metadata()


def test_write_data(args_dict_site, io_handler):

    # test writer with no workflow configuration
    w_1 = writer.ModelDataWriter()
    with pytest.raises(
        AttributeError, match=r"\'NoneType\' object has no attribute \'product_data_file_name\'"
    ):
        w_1.write_data(None)

    # test writer with workflow configuration initialised, but not set
    c_2 = workflow.WorkflowDescription(label="unit-test", args_dict=args_dict_site)
    w_2 = writer.ModelDataWriter(c_2)
    with pytest.raises(
        AttributeError, match=re.escape("'NoneType' object has no attribute 'write'")
    ):
        w_2.write_data(None)

    # TODO tests which actually write data
