#!/usr/bin/python3

import logging
import unittest

import astropy.units as u

from simtools.simtel.simtel_runner_array import SimtelRunnerArray
from simtools.model.array_model import ArrayModel

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class TestSimtelRunnerRayTracing(unittest.TestCase):

    def setUp(self):
        arrayConfigData = {
            'site': 'North',
            'layoutName': 'Prod5',
            'modelVersion': 'Prod5',
            'default': {
                'LST': '1',
                'MST': 'FlashCam-D'
            },
            'M-05': 'NectarCam-D'
        }
        self.aarrayModel = ArrayModel(label='test', arrayConfigData=arrayConfigData)

        self.simtelRunner = SimtelRunnerArray(
            arrayModel=self.arrayModel,
            configData={
                'zenithAngle': 20 * u.deg,
                'azimuthAngle': 0 * u.deg
            }
        )

    # def test_run(self):
    #     self.simtelRunner.run(test=True, force=True)


if __name__ == '__main__':
    unittest.main()
