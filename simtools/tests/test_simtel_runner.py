#!/usr/bin/python3

import logging
import astropy.units as u

import simtools.config as cfg
from simtools.simtel_runner import SimtelRunner
from simtools.model.telescope_model import TelescopeModel

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
        zenithAngle=20 * u.deg,
        offAxisAngle=2 * u.deg,
        sourceDistance=12 * u.km
    )

    logging.info(simtel)
    # simtel.run(test=True, force=True)


if __name__ == '__main__':

    test_ray_tracing_mode()
