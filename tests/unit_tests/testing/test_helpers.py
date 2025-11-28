#!/usr/bin/python3

import os
from unittest import mock

import pytest

from simtools.testing import helpers

SKIPPING_TEST = "Skipping test not meant for multiple model versions."


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


def test_skip_camera_efficiency_old_testeff(new_testeff_version):
    config = {"application": "camera-efficiency", "test_name": "some_test"}
    with mock.patch(new_testeff_version, return_value=False):
        skip_string = helpers.skip_camera_efficiency(config)
        assert "old testeff" in skip_string


def test_skip_camera_efficiency_new_testeff(new_testeff_version):
    config = {"application": "camera-efficiency", "test_name": "some_test"}
    with mock.patch(new_testeff_version, return_value=True):
        helpers.skip_camera_efficiency(config)


def test_skip_camera_efficiency_not_camera_efficiency():
    config = {"application": "other-application", "test_name": "some_test"}
    helpers.skip_camera_efficiency(config)


def test_new_testeff_version_file_not_found(builtins_open):
    with mock.patch.dict(os.environ, {"SIMTOOLS_SIMTEL_PATH": "/fake_for_test/path"}):
        with mock.patch(builtins_open, side_effect=FileNotFoundError):
            with pytest.raises(
                FileNotFoundError, match=r"The testeff executable could not be found."
            ):
                helpers._new_testeff_version()


def test_skip_multiple_version_test_single_version():
    config = {"configuration": {"model_version": "5.0.0"}}
    model_version = ["5.0.0"]
    result = helpers.skip_multiple_version_test(config, model_version)
    assert result is None


def test_skip_multiple_version_test_matching_multiple_versions():
    config = {"configuration": {"model_version": ["5.0.0", "6.0.0"]}}
    model_version = ["5.0.0", "6.0.0"]
    result = helpers.skip_multiple_version_test(config, model_version)
    assert result is None


def test_skip_multiple_version_test_mismatched_multiple_versions():
    config = {"configuration": {"model_version": ["5.0.0"]}}
    model_version = ["5.0.0", "6.0.0"]
    result = helpers.skip_multiple_version_test(config, model_version)
    assert result == SKIPPING_TEST


def test_skip_multiple_version_test_no_model_version_in_config():
    config = {"configuration": {}}
    model_version = ["5.0.0", "6.0.0"]
    result = helpers.skip_multiple_version_test(config, model_version)
    assert result == SKIPPING_TEST


def test_skip_multiple_version_test_empty_model_version_list():
    config = {"configuration": {"model_version": []}}
    model_version = ["5.0.0", "6.0.0"]
    result = helpers.skip_multiple_version_test(config, model_version)
    assert result == SKIPPING_TEST


def test_skip_multiple_version_test_config_model_version_not_list():
    config = {"configuration": {"model_version": "5.0.0"}}
    model_version = ["5.0.0", "6.0.0"]
    result = helpers.skip_multiple_version_test(config, model_version)
    assert result == SKIPPING_TEST


def test_skip_multiple_version_test_no_configuration_key():
    config = {}
    model_version = ["5.0.0", "6.0.0"]
    result = helpers.skip_multiple_version_test(config, model_version)
    assert result == SKIPPING_TEST


def test_skip_multiple_version_test_single_model_version_no_config_model_version():
    config = {"configuration": {}}
    model_version = ["5.0.0"]
    result = helpers.skip_multiple_version_test(config, model_version)
    assert result is None


def test_skip_multiple_version_test_empty_model_version():
    config = {"configuration": {"model_version": []}}
    model_version = []
    result = helpers.skip_multiple_version_test(config, model_version)
    assert result is None


def test_skip_multiple_version_test_model_version_not_a_list():
    config = {"configuration": {"model_version": "5.0.0"}}
    model_version = "5.0.0"
    result = helpers.skip_multiple_version_test(config, model_version)
    assert result is None


def test_skip_multiple_version_test_model_version_not_list():
    config = {"configuration": {"model_version": ["5.0.0", "6.0.0"]}}
    model_version = "5.0.0"  # Not a list
    result = helpers.skip_multiple_version_test(config, model_version)
    assert result is None
