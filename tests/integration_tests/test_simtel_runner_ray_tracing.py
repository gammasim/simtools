#!/usr/bin/python3

import pytest
import logging

import astropy.units as u

from simtools.simtel.simtel_runner_ray_tracing import SimtelRunnerRayTracing
from simtools.model.telescope_model import TelescopeModel


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def telescopeModel(cfg_setup, set_db):
    telescopeModel = TelescopeModel(
        site="north",
        telescopeModelName="lst-1",
        modelVersion="Current",
        label="test-simtel",
    )
    return telescopeModel


@pytest.fixture
def simtelRunner(telescopeModel):
    simtelRunner = SimtelRunnerRayTracing(
        telescopeModel=telescopeModel,
        configData={
            "zenithAngle": 20 * u.deg,
            "offAxisAngle": 2 * u.deg,
            "sourceDistance": 12 * u.km,
        },
    )
    return simtelRunner


def test_run(set_simtelarray, simtelRunner):
    simtelRunner.run(test=True, force=True)
