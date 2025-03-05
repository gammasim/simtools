#!/usr/bin/python3

from pathlib import Path

import simtools.constants as constants


def test_constants():
    assert isinstance(constants.METADATA_JSON_SCHEMA, Path)
    assert constants.METADATA_JSON_SCHEMA.is_file()

    assert isinstance(constants.MODEL_PARAMETER_SCHEMA_PATH, Path)
    assert constants.MODEL_PARAMETER_SCHEMA_PATH.is_dir()
