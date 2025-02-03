#!/usr/bin/python3

import os
from unittest import mock

import pytest

from simtools.testing import helpers


@pytest.fixture
def new_testeff_version():
    return "simtools.testing.helpers._new_testeff_version"


@pytest.fixture
def builtins_open():
    return "builtins.open"


def test_new_testeff_version_true(builtins_open):
    with mock.patch.dict(os.environ, {"SIMTOOLS_SIMTEL_PATH": "/fake/path"}):
        with mock.patch(
            builtins_open,
            mock.mock_open(
                read_data="/* Combine the include paths such that those from '-I...' options */"
            ),
        ):
            assert helpers._new_testeff_version() is True


def test_new_testeff_version_false(builtins_open):
    with mock.patch.dict(os.environ, {"SIMTOOLS_SIMTEL_PATH": "/fake/path"}):
        with mock.patch(builtins_open, mock.mock_open(read_data="Some other content")):
            assert helpers._new_testeff_version() is False


def test_new_testeff_version_no_env_var():
    if "SIMTOOLS_SIMTEL_PATH" in os.environ:
        del os.environ["SIMTOOLS_SIMTEL_PATH"]
    with pytest.raises(TypeError):
        helpers._new_testeff_version()


def test_skip_camera_efficiency_old_testeff(new_testeff_version):
    config = {"APPLICATION": "camera-efficiency", "TEST_NAME": "some_test"}
    with mock.patch(new_testeff_version, return_value=False):
        skip_string = helpers.skip_camera_efficiency(config)
        assert "old testeff" in skip_string


def test_skip_camera_efficiency_new_testeff(new_testeff_version):
    config = {"APPLICATION": "camera-efficiency", "TEST_NAME": "some_test"}
    with mock.patch(new_testeff_version, return_value=True):
        helpers.skip_camera_efficiency(config)


def test_skip_camera_efficiency_not_camera_efficiency():
    config = {"APPLICATION": "other-application", "TEST_NAME": "some_test"}
    helpers.skip_camera_efficiency(config)


def test_new_testeff_version_file_not_found(builtins_open):
    with mock.patch.dict(os.environ, {"SIMTOOLS_SIMTEL_PATH": "/fake_for_test/path"}):
        with mock.patch(builtins_open, side_effect=FileNotFoundError):
            with pytest.raises(
                FileNotFoundError, match="The testeff executable could not be found."
            ):
                helpers._new_testeff_version()
