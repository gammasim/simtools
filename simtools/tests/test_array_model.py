#!/usr/bin/python3

import logging

from simtools.model.array_model import ArrayModel

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_input_validation():

    arrayConfigData = {
        'site': 'South',
        'arrayName': 'Prod5',
        'default': {
            'LST': 'D',
            'MST': 'FlashCam-D',
            'SST': 'D'
        },
        'L-01': '1'
    }
    am = ArrayModel(arrayConfigData=arrayConfigData)


if __name__ == '__main__':
    test_input_validation()
