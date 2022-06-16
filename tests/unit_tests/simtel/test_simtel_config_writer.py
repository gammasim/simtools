#!/usr/bin/python3

import pytest
import logging

import simtools.io_handler as io
from simtools.util.general import fileHasText
from simtools.simtel.simtel_config_writer import SimtelConfigWriter
from simtools.model.telescope_model import TelescopeModel
from simtools.layout.layout_array import LayoutArray


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def telescopeModel():
    telescopeModel = TelescopeModel(
        site="North",
        telescopeModelName="LST-1",
        modelVersion="Current",
        label="test-telescope-model",
    )
    return telescopeModel


@pytest.fixture
def simtelConfigWriter(cfg_setup, set_db):

    simtelConfigWriter = SimtelConfigWriter(
        site="North",
        modelVersion="Current",
        label="test-simtel-config-writer",
        telescopeModelName="TestTelecope",
    )
    return simtelConfigWriter


@pytest.fixture
def layout():
    layout = LayoutArray.fromLayoutArrayName("South-4LST")
    return layout


# @pytest.mark.skip(reason="TODO :test_write_array_config_file - KeyError: 'Current'")
def test_write_array_config_file(simtelConfigWriter, layout, telescopeModel):
    file = io.getTestOutputFile("simtel-config-writer_array.txt")
    simtelConfigWriter.writeArrayConfigFile(
        configFilePath=file,
        layout=layout,
        telescopeModel=[telescopeModel] * 4,
        siteParameters={},
    )
    assert fileHasText(file, "TELESCOPE == 1")


def test_write_tel_config_file(simtelConfigWriter):
    file = io.getTestOutputFile("simtel-config-writer_telescope.txt")
    simtelConfigWriter.writeTelescopeConfigFile(
        configFilePath=file, parameters={"par": {"Value": 1}}
    )
    assert fileHasText(file, "par = 1")
