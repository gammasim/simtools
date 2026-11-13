#!/usr/bin/python3

from pathlib import Path

from simtools import constants


def test_versioned_test_resources_accepts_checkout_and_parent(tmp_test_directory):
    checkout = Path(tmp_test_directory) / "simtools-tests"
    resource = checkout / "v1.0.0" / "integration_tests"
    resource.mkdir(parents=True)
    for configured in (checkout, checkout.parent):
        assert constants._versioned_test_resources_path(configured, "v1.0.0") == resource
