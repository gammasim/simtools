#!/usr/bin/python3

import logging
import pytest

from simtools.simtel.simtel_runner import SimtelRunner, SimtelExecutionError

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def simtelRunner(cfg_setup):

    simtelRunner = SimtelRunner()
    return simtelRunner


def test_run(simtelRunner):
    with pytest.raises(RuntimeError):
        simtelRunner.run()


def test_simtel_execution_error(simtelRunner):
    with pytest.raises(SimtelExecutionError):
        simtelRunner._raiseSimtelError()
