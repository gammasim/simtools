#!/usr/bin/python3

import logging

import pytest

from simtools.layout.layout_array import LayoutArray
from simtools.model.telescope_model import TelescopeModel
from simtools.simtel.simtel_config_writer import SimtelConfigWriter
from simtools.util.general import file_has_text

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def telescopeModel(db_config, io_handler):
    telescopeModel = TelescopeModel(
        site="North",
        telescopeModelName="LST-1",
        modelVersion="Current",
        label="test-telescope-model",
        mongoDBConfig=db_config,
    )
    return telescopeModel


@pytest.fixture
def simtelConfigWriter():

    simtelConfigWriter = SimtelConfigWriter(
        site="North",
        modelVersion="Current",
        label="test-simtel-config-writer",
        telescopeModelName="TestTelecope",
    )
    return simtelConfigWriter


@pytest.fixture
def layout(io_handler):
    layout = LayoutArray.from_layout_array_name("South-4LST")
    return layout


# @pytest.mark.skip(reason="TODO :test_write_array_config_file - KeyError: 'Current'")
def test_write_array_config_file(simtelConfigWriter, layout, telescopeModel, io_handler):
    file = io_handler.get_output_file(fileName="simtel-config-writer_array.txt", test=True)
    simtelConfigWriter.write_array_config_file(
        configFilePath=file,
        layout=layout,
        telescopeModel=[telescopeModel] * 4,
        siteParameters={},
    )
    assert file_has_text(file, "TELESCOPE == 1")


def test_write_tel_config_file(simtelConfigWriter, io_handler):
    file = io_handler.get_output_file(fileName="simtel-config-writer_telescope.txt", test=True)
    simtelConfigWriter.write_telescope_config_file(
        configFilePath=file, parameters={"par": {"Value": 1}}
    )
    assert file_has_text(file, "par = 1")
