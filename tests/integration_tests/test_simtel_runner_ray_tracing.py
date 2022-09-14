#!/usr/bin/python3

import logging

import astropy.units as u
import pytest

from simtools.model.telescope_model import TelescopeModel
from simtools.simtel.simtel_runner_ray_tracing import SimtelRunnerRayTracing

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def telescopeModel(set_simtools):
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
