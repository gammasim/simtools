#!/usr/bin/python3

import logging

import pytest

from simtools.simtel.simtel_runner import SimtelExecutionError, SimtelRunner

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def simtel_runner(simtel_path):

    simtel_runner = SimtelRunner(simtel_source_path=simtel_path)
    return simtel_runner


def test_run(simtel_runner):
    with pytest.raises(RuntimeError):
        simtel_runner.run()


def test_simtel_execution_error(simtel_runner):
    with pytest.raises(SimtelExecutionError):
        simtel_runner._raise_simtel_error()
