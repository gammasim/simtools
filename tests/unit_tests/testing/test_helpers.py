#!/usr/bin/python3

from simtools.testing import helpers

SKIPPING_TEST = "Skipping test not meant for multiple model versions."


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


def test_skip_multiple_version_test_model_version_not_a_list():
    config = {"configuration": {"model_version": "5.0.0"}}
    model_version = "5.0.0"
    result = helpers.skip_multiple_version_test(config, model_version)
    assert result is None
