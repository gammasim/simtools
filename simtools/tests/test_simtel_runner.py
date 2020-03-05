#!/usr/bin/python3

import logging

from simtools.util import config as cfg
from simtools.simtel_runner import SimtelRunner
from simtools.telescope_model import TelescopeModel

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)

config = cfg.loadConfig()


def test_ray_tracing_mode():
    tel = TelescopeModel(
        yamlDBPath=config['yamlDBPath'],
        telescopeType='lst',
        site='south',
        version='prod4',
        label='test-simtel'
    )

    simtel = SimtelRunner(
        simtelSourcePath=config['simtelPath'],
        mode='ray-tracing',
        telescopeModel=tel,
        zenithAngle=20,
        offAxisAngle=2
    )

    logger.info(simtel)

    simtel.run(test=True, force=True)


if __name__ == '__main__':

    test_ray_tracing_mode()
