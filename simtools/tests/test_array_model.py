#!/usr/bin/python3

import logging

from simtools.model.array_model import ArrayModel

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_input_validation():
    arrayConfigData = {
        'site': 'North',
        'arrayName': 'Prod5',
        'modelVersion': 'Prod5',
        'default': {
            'LST': '1',
            'MST': 'FlashCam-D'
        },
        'M-05': 'NectarCam-D'
    }
    am = ArrayModel(label='test', arrayConfigData=arrayConfigData)

    am.printTelescopeList()


def test_exporting_config_files():
    arrayConfigData = {
        'site': 'North',
        'arrayName': 'Prod5',
        'modelVersion': 'Prod5',
        'default': {
            'LST': '1',
            'MST': 'FlashCam-D'
        },
        'M-05': {
            'name': 'NectarCam-D',
            'fadc_pulse_shape': 'Pulse_template_nectarCam_17042020-noshift.dat',
            'discriminator_pulse_shape': 'Pulse_template_nectarCam_17042020-noshift.dat'
        }
    }
    am = ArrayModel(label='test', arrayConfigData=arrayConfigData)

    am.exportSimtelTelescopeConfigFiles()
    am.exportSimtelArrayConfigFile()


if __name__ == '__main__':
    # test_input_validation()
    test_exporting_config_files()
