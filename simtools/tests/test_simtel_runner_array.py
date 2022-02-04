#!/usr/bin/python3

import logging
import unittest
from pathlib import Path

import astropy.units as u

import simtools.io_handler as io
from simtools.simtel.simtel_runner_array import SimtelRunnerArray
from simtools.model.array_model import ArrayModel

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class TestSimtelRunnerArray(unittest.TestCase):
    def setUp(self):
        arrayConfigData = {
            "site": "North",
            "layoutName": "1LST",
            "modelVersion": "Prod5",
            "default": {"LST": "1"},
        }
        self.arrayModel = ArrayModel(
            label="test-lst-array", arrayConfigData=arrayConfigData
        )

        self.simtelRunner = SimtelRunnerArray(
            arrayModel=self.arrayModel,
            configData={
                "primary": "proton",
                "zenithAngle": 20 * u.deg,
                "azimuthAngle": 0 * u.deg,
            },
        )
        self.corsikaFile = io.getTestDataFile(
            "run1_proton_za20deg_azm0deg-North-1LST_trigger_rates.corsika.zst"
        )

    def test_run(self):
        self.simtelRunner.run(test=False, force=True, inputFile=self.corsikaFile, run=1)

    def test_run_script(self):
        script = self.simtelRunner.getRunScript(run=1, inputFile=self.corsikaFile)
        assert Path(script).exists()


if __name__ == "__main__":
    unittest.main()
