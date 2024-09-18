#!/usr/bin/python3

import logging
from pathlib import Path

import pytest

from simtools.testing import assertions

logging.getLogger().setLevel(logging.DEBUG)


@pytest.fixture()
def test_json_file():
    return Path("tests/resources/reference_point_altitude.json")


@pytest.fixture()
def test_yaml_file():
    return Path("tests/resources/num_gains.schema.yml")


def test_assert_file_type_json(test_json_file, test_yaml_file):

    assert assertions.assert_file_type("json", test_json_file)
    assert not assertions.assert_file_type("json", "tests/resources/does_not_exist.json")
    assert not assertions.assert_file_type("json", test_yaml_file)

    assert assertions.assert_file_type("json", Path(test_json_file))


def test_assert_file_type_yaml(test_json_file, test_yaml_file, caplog):

    assert assertions.assert_file_type("yaml", test_yaml_file)
    assert assertions.assert_file_type("yml", test_yaml_file)
    assert not assertions.assert_file_type("yml", "tests/resources/does_not_exit.schema.yml")

    assert not assertions.assert_file_type(
        "yaml", "tests/resources/telescope_positions-South-ground.ecsv"
    )


def test_assert_file_type_others(caplog):

    with caplog.at_level(logging.INFO):
        assert assertions.assert_file_type(
            "ecsv", "tests/resources/telescope_positions-South-ground.ecsv"
        )
    assert (
        "File type test is checking suffix only for tests/resources/"
        "telescope_positions-South-ground.ecsv (suffix: ecsv)" in caplog.text
    )
