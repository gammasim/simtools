#!/usr/bin/python3

import logging

import pytest

from simtools.layout.layout_array import LayoutArray
from simtools.model.telescope_model import TelescopeModel
from simtools.simtel.simtel_config_writer import SimtelConfigWriter
from simtools.util.general import fileHasText

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def telescopeModel(db_connection, io_handler):
    telescopeModel = TelescopeModel(
        site="North",
        telescopeModelName="LST-1",
        modelVersion="Current",
        label="test-telescope-model",
        mongoDBConfigFile=str(db_connection),
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
    layout = LayoutArray.fromLayoutArrayName("South-4LST")
    return layout


# @pytest.mark.skip(reason="TODO :test_write_array_config_file - KeyError: 'Current'")
def test_write_array_config_file(simtelConfigWriter, layout, telescopeModel, io_handler):
    file = io_handler.getOutputFile(fileName="simtel-config-writer_array.txt", test=True)
    simtelConfigWriter.writeArrayConfigFile(
        configFilePath=file,
        layout=layout,
        telescopeModel=[telescopeModel] * 4,
        siteParameters={},
    )
    assert fileHasText(file, "TELESCOPE == 1")


def test_write_tel_config_file(simtelConfigWriter, io_handler):
    file = io_handler.getOutputFile(fileName="simtel-config-writer_telescope.txt", test=True)
    simtelConfigWriter.writeTelescopeConfigFile(
        configFilePath=file, parameters={"par": {"Value": 1}}
    )
    assert fileHasText(file, "par = 1")
