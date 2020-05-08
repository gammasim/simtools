#!/usr/bin/python3

import logging
import astropy.units as u

from simtools.simtel_runner import SimtelRunner, SimtelExecutionError
from simtools.model.telescope_model import TelescopeModel
import simtools.config as cfg

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_ray_tracing_mode():
    tel = TelescopeModel(
        telescopeType='lst',
        site='south',
        version='prod4',
        label='test-simtel'
    )

    simtel = SimtelRunner(
        mode='ray-tracing',
        telescopeModel=tel,
        zenithAngle=20 * u.deg,
        offAxisAngle=2 * u.deg,
        sourceDistance=12 * u.km
    )

    logger.info(simtel)
    # simtel.run(test=True, force=True)


def test_catching_model_error():
    tel = TelescopeModel(
        telescopeType='mst-flashcam',
        site='south',
        version='prod4',
        label='test-simtel'
    )

    # Adding a invalid parameter
    tel.addParameters(invalid_parameter='invalid_value')

    simtel = SimtelRunner(
        mode='ray-tracing',
        telescopeModel=tel,
        zenithAngle=20 * u.deg,
        offAxisAngle=0 * u.deg,
        sourceDistance=12 * u.km
    )

    logger.info(simtel)
    try:
        simtel.run(test=True, force=True)
    except SimtelExecutionError:
        logger.info('Error catch properly - everything seems fine')


if __name__ == '__main__':

    test_ray_tracing_mode()
    test_catching_model_error()
