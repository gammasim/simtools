#!/usr/bin/python3

import logging
import astropy.units as u

from simtools.simtel.simtel_runner_ray_tracing import SimtelRunnerRayTracing
from simtools.model.telescope_model import TelescopeModel

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_ray_tracing():
    tel = TelescopeModel(
        site='north',
        telescopeModelName='lst-1',
        modelVersion='Current',
        label='test-simtel'
    )

    simtel = SimtelRunnerRayTracing(
        telescopeModel=tel,
        configData={
            'zenithAngle': 20 * u.deg,
            'offAxisAngle': 2 * u.deg,
            'sourceDistance': 12 * u.km
        }
    )

    logger.info(simtel)
    # simtel.run(test=True, force=True)


if __name__ == '__main__':

    test_ray_tracing()
