#!/usr/bin/python3

import logging

from simtools.model.telescope_model import TelescopeModel
from simtools.camera_efficiency import CameraEfficiency

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_simulate():
    tel = TelescopeModel(
        telescopeType='lst',
        site='south',
        version='p3',
        label='test_camera_eff'
    )
    ce = CameraEfficiency(telescopeModel=tel)
    ce.simulate()


if __name__ == '__main__':

    test_simulate()
    pass
