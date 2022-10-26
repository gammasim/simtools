#!/usr/bin/python3

import logging
import re
from pathlib import Path

import pytest

import simtools.util.model_data_writer as writer
import simtools.util.workflow_description as workflow

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_write_metadata(args_dict_site, io_handler, tmp_test_directory):

    # test writer of metadata
    w_1 = writer.ModelDataWriter(args_dict=args_dict_site)
    yml_file = w_1.write_metadata()
    assert Path(yml_file).exists()

    yml_file = w_1.write_metadata(ymlfile=str(tmp_test_directory) + "/test_file.yml")
    assert Path(yml_file).exists()

    with pytest.raises(FileNotFoundError):
        w_1.write_metadata(ymlfile="./this_directory_is_not_theta/test_file.yml")


def test_initialize():
    # test writer with no workflow configuration and no configuration dictionary defined
    with pytest.raises(TypeError, match=r"\'NoneType\' object is not subscriptable"):
        writer.ModelDataWriter(workflow_config=None, args_dict=None)


def test_write_data(args_dict_site, io_handler):
    # test writer with workflow configuration initialised, but not set
    c_2 = workflow.WorkflowDescription(args_dict=args_dict_site)
    w_2 = writer.ModelDataWriter(c_2)
    with pytest.raises(
        AttributeError, match=re.escape("'NoneType' object has no attribute 'write'")
    ):
        w_2.write_data(None)

    # TODO tests which actually write data
