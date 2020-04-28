#!/usr/bin/python3

import logging
from astropy import units

from simtools.util import config as cfg
from simtools.simtel_runner import SimtelRunner
from simtools.telescope_model import TelescopeModel

logging.getLogger().setLevel(logging.DEBUG)


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
        zenithAngle=20 * units.deg,
        offAxisAngle=2,
        sourceDistance=10
    )

    logging.info(simtel)
    # simtel.run(test=True, force=True)


if __name__ == '__main__':

    test_ray_tracing_mode()
