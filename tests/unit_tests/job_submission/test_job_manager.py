import logging
import pytest

import simtools.job_submission.job_manager as jm

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def label():
    return "test-shower-simulator"


def test_test_submission_system(label):
    jm.JobManager(submitCommand=None)
    jm.JobManager(submitCommand="local")
    with pytest.raises(jm.MissingWorkloadManager):
        jm.JobManager(submitCommand='abc')
