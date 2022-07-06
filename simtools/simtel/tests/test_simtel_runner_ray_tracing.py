#!/usr/bin/python3

import pytest
import logging
import unittest

import astropy.units as u

from simtools.simtel.simtel_runner_ray_tracing import SimtelRunnerRayTracing
from simtools.model.telescope_model import TelescopeModel
from simtools.util.tests import (
    has_db_connection,
    simtel_installed,
    DB_CONNECTION_MSG,
    SIMTEL_MSG,
)


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class TestSimtelRunnerRayTracing(unittest.TestCase):
    def setUp(self):
        self.telescopeModel = TelescopeModel(
            site="north",
            telescopeModelName="lst-1",
            modelVersion="Current",
            label="test-simtel",
        )

        self.simtelRunner = SimtelRunnerRayTracing(
            telescopeModel=self.telescopeModel,
            configData={
                "zenithAngle": 20 * u.deg,
                "offAxisAngle": 2 * u.deg,
                "sourceDistance": 12 * u.km,
            },
        )

    @pytest.mark.skipif(not has_db_connection(), reason=DB_CONNECTION_MSG)
    @pytest.mark.skipif(not simtel_installed(), reason=SIMTEL_MSG)
    def test_run(self):
        self.simtelRunner.run(test=True, force=True)


if __name__ == "__main__":
    unittest.main()
