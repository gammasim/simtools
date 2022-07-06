#!/usr/bin/python3

import pytest
import logging

from simtools.model.array_model import ArrayModel
from simtools.util.tests import (
    has_db_connection,
    has_config_file,
    DB_CONNECTION_MSG,
    CONFIG_FILE_MSG,
)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.mark.skipif(not has_config_file(), reason=CONFIG_FILE_MSG)
def test_input_validation():
    arrayConfigData = {
        "site": "North",
        "layoutName": "Prod5",
        "modelVersion": "Prod5",
        "default": {"LST": "1", "MST": "FlashCam-D"},
        "M-05": "NectarCam-D",
    }
    am = ArrayModel(label="test", arrayConfigData=arrayConfigData)

    am.printTelescopeList()


@pytest.mark.skipif(not has_db_connection(), reason=DB_CONNECTION_MSG)
def test_exporting_config_files():
    arrayConfigData = {
        "site": "North",
        "layoutName": "Prod5",
        "modelVersion": "Prod5",
        "default": {"LST": "1", "MST": "FlashCam-D"},
        "M-05": {
            "name": "NectarCam-D",
            "fadc_pulse_shape": "Pulse_template_nectarCam_17042020-noshift.dat",
            "discriminator_pulse_shape": "Pulse_template_nectarCam_17042020-noshift.dat",
        },
    }
    am = ArrayModel(label="test", arrayConfigData=arrayConfigData)

    am.exportSimtelTelescopeConfigFiles()
    am.exportSimtelArrayConfigFile()
