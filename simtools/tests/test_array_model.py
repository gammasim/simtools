#!/usr/bin/python3

import logging

from simtools.model.telescope_model import ArrayModel

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_input_validation():

    # Dict with the array config data
    arrayConfigData = {
        'site': 'South',
        'layoutName': 'Prod5',
        'default': {
            'LST': 'D',
            'MST': 'FlashCam-D',
            'SST': 'D'
        },
        'L-01': '1'
    }

    am = ArrayModel(label='test', arrayConfigData=arrayConfigData)

    return


if __name__ == '__main__':

    pass
