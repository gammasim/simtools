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
    am = ArrayModel(arrayConfigData=arrayConfigData)


if __name__ == '__main__':
    test_input_validation()
