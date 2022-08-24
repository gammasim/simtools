#!/usr/bin/python3

import logging

import pytest

from simtools.util import names

logging.getLogger().setLevel(logging.DEBUG)


def test_validate_telescope_names():

    telescopes = {"sst-d": "SST-D", "mst-flashcam-d": "MST-FlashCam-D", "sct-d": "SCT-D"}

    for key, value in telescopes.items():
        logging.getLogger().info("Validating {}".format(key))
        newName = names.validateTelescopeModelName(key)
        logging.getLogger().info("New name {}".format(newName))

        assert value == newName


def test_validate_other_names():
    modelVersion = names.validateModelVersionName("p4")
    logging.getLogger().info(modelVersion)

    assert modelVersion == "prod4"


def test_simtoolsInstrumentName():

    assert names.simtoolsInstrumentName("South", "MST", "FlashCam", "D") == "South-MST-FlashCam-D"
    assert names.simtoolsInstrumentName("North", "MST", "NectarCam", "7") == "North-MST-NectarCam-7"

    with pytest.raises(ValueError):
        names.simtoolsInstrumentName("West", "MST", "FlashCam", "D")
