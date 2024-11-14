#!/usr/bin/python3

import os
from unittest import mock

import pytest

from simtools.testing import helpers


@pytest.fixture
def new_testeff_version():
    return "simtools.testing.helpers._new_testeff_version"


def test_new_testeff_version_true():
    with mock.patch.dict(os.environ, {"SIMTOOLS_SIMTEL_PATH": "/fake/path"}):
        with mock.patch(
            "builtins.open",
            mock.mock_open(
                read_data="/* Combine the include paths such that those from '-I...' options */"
            ),
        ):
            assert helpers._new_testeff_version() is True


def test_new_testeff_version_false():
    with mock.patch.dict(os.environ, {"SIMTOOLS_SIMTEL_PATH": "/fake/path"}):
        with mock.patch("builtins.open", mock.mock_open(read_data="Some other content")):
            assert helpers._new_testeff_version() is False


def test_new_testeff_version_no_env_var():
    if "SIMTOOLS_SIMTEL_PATH" in os.environ:
        del os.environ["SIMTOOLS_SIMTEL_PATH"]
    with pytest.raises(TypeError):
        helpers._new_testeff_version()


def test_skip_camera_efficiency_old_testeff(new_testeff_version):
    config = {"APPLICATION": "camera-efficiency", "TEST_NAME": "some_test"}
    with mock.patch(new_testeff_version, return_value=False):
        with pytest.raises(pytest.skip.Exception):
            helpers.skip_camera_efficiency(config)


def test_skip_camera_efficiency_new_testeff(new_testeff_version):
    config = {"APPLICATION": "camera-efficiency", "TEST_NAME": "some_test"}
    with mock.patch(new_testeff_version, return_value=True):
        helpers.skip_camera_efficiency(config)


def test_skip_camera_efficiency_specific_test(new_testeff_version):
    config = {
        "APPLICATION": "simtools-validate-camera-efficiency",
        "TEST_NAME": "SSTS",
    }
    with mock.patch(new_testeff_version, return_value=True):
        with pytest.raises(pytest.skip.Exception):
            helpers.skip_camera_efficiency(config)


def test_skip_camera_efficiency_not_camera_efficiency():
    config = {"APPLICATION": "other-application", "TEST_NAME": "some_test"}
    helpers.skip_camera_efficiency(config)
