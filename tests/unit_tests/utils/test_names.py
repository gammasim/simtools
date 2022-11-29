#!/usr/bin/python3

import logging

import pytest

from simtools.util import names

logging.getLogger().setLevel(logging.DEBUG)


def test_validate_telescope_name():

    telescopes = {"sst-d": "SST-D", "mst-flashcam-d": "MST-FlashCam-D", "sct-d": "SCT-D"}

    for key, value in telescopes.items():
        logging.getLogger().info(f"Validating {key}")
        new_name = names.validate_telescope_model_name(key)
        logging.getLogger().info(f"New name {new_name}")

        assert value == new_name


def test_validate_telescope_name_db():

    telescopes = {
        "south-sst-d": "South-SST-D",
        "north-mst-nectarcam-d": "North-MST-NectarCam-D",
        "north-lst-1": "North-LST-1",
    }

    for key, value in telescopes.items():
        logging.getLogger().info(f"Validating {key}")
        new_name = names.validate_telescope_name_db(key)
        logging.getLogger().info(f"New name {new_name}")

        assert value == new_name

    telescopes = {
        "ssss-sst-d": "SSSS-SST-D",
        "no-rth-mst-nectarcam-d": "No-rth-MST-NectarCam-D",
        "north-ls-1": "North-LS-1",
    }

    for key, value in telescopes.items():
        logging.getLogger().info(f"Validating {key}")
        with pytest.raises(ValueError):
            names.validate_telescope_name_db(key)


def test_validate_other_names():
    model_version = names.validate_model_version_name("p4")
    logging.getLogger().info(model_version)

    assert model_version == "prod4"


def test_simtools_instrument_name():

    assert names.simtools_instrument_name("South", "MST", "FlashCam", "D") == "South-MST-FlashCam-D"
    assert (
        names.simtools_instrument_name("North", "MST", "NectarCam", "7") == "North-MST-NectarCam-7"
    )

    with pytest.raises(ValueError):
        names.simtools_instrument_name("West", "MST", "FlashCam", "D")
