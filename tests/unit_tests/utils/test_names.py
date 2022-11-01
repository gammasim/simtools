#!/usr/bin/python3

import logging

import pytest

from simtools.util import names

logging.getLogger().setLevel(logging.DEBUG)


def test_validate_telescope_name():

    telescopes = {"sst-d": "SST-D", "mst-flashcam-d": "MST-FlashCam-D", "sct-d": "SCT-D"}

    for key, value in telescopes.items():
        logging.getLogger().info("Validating {}".format(key))
        newName = names.validate_telescope_model_name(key)
        logging.getLogger().info("New name {}".format(newName))

        assert value == newName


def test_validate_telescope_name_db():

    telescopes = {
        "south-sst-d": "South-SST-D",
        "north-mst-nectarcam-d": "North-MST-NectarCam-D",
        "north-lst-1": "North-LST-1",
    }

    for key, value in telescopes.items():
        logging.getLogger().info("Validating {}".format(key))
        newName = names.validate_telescope_name_db(key)
        logging.getLogger().info("New name {}".format(newName))

        assert value == newName

    telescopes = {
        "ssss-sst-d": "SSSS-SST-D",
        "no-rth-mst-nectarcam-d": "No-rth-MST-NectarCam-D",
        "north-ls-1": "North-LS-1",
    }

    for key, value in telescopes.items():
        logging.getLogger().info("Validating {}".format(key))
        with pytest.raises(ValueError):
            names.validate_telescope_name_db(key)


def test_validate_other_names():
    modelVersion = names.validate_model_version_name("p4")
    logging.getLogger().info(modelVersion)

    assert modelVersion == "prod4"


def test_simtools_instrument_name():

    assert names.simtools_instrument_name("South", "MST", "FlashCam", "D") == "South-MST-FlashCam-D"
    assert (
        names.simtools_instrument_name("North", "MST", "NectarCam", "7") == "North-MST-NectarCam-7"
    )

    with pytest.raises(ValueError):
        names.simtools_instrument_name("West", "MST", "FlashCam", "D")
