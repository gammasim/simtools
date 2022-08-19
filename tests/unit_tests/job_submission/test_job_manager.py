import logging
import pytest

import simtools.job_submission.job_manager as jm

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def label():
    return "test-shower-simulator"


def test_test_submission_system(label):
    job_manager_1 = jm.JobManager(submitCommand=None)
    job_manager_2 = jm.JobManager(submitCommand="seriell_script")
    with pytest.raises(jm.MissingWorkloadManager):
        job_manager_fail_1 = jm.JobManager(submitCommand='abc')
