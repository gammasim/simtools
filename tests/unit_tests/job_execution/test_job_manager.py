import logging

import pytest

import simtools.job_execution.job_manager as jm

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def label():
    return "test-shower-simulator"


def test_test_submission_system(label):
    jm.JobManager(submit_command=None)
    jm.JobManager(submit_command="local")
    with pytest.raises(jm.MissingWorkloadManager):
        jm.JobManager(submit_command="abc")
