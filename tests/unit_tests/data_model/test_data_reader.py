#!/usr/bin/python3

import json
import logging

import astropy.units as u
import jsonschema
import pytest
from astropy.io.registry.base import IORegistryError
from astropy.table import Table

import simtools.utils.general as gen
from simtools.data_model import data_reader

logger = logging.getLogger()


def test_read_table_from_file(telescope_north_test_file):
    assert isinstance(
        data_reader.read_table_from_file(telescope_north_test_file),
        Table,
    )

    with pytest.raises(FileNotFoundError):
        data_reader.read_table_from_file("non_existing_file.fits")

    with pytest.raises(IORegistryError):
        data_reader.read_table_from_file(None)


def test_read_table_from_file_and_validate(telescope_north_test_file):
    # schema file from metadata in table
    assert isinstance(
        data_reader.read_table_from_file(telescope_north_test_file, validate=True),
        Table,
    )

    assert isinstance(
        data_reader.read_table_from_file(
            "tests/resources/telescope_positions-North-utm-without-cta-meta.ecsv",
            validate=True,
            metadata_file="tests/resources/telescope_positions-North-utm.meta.yml",
        ),
        Table,
    )


def test_read_value_from_file(tmp_test_directory):
    assert isinstance(
        data_reader.read_value_from_file("tests/resources/reference_point_altitude.json"),
        u.quantity.Quantity,
    )

    with pytest.raises(FileNotFoundError):
        data_reader.read_value_from_file("this_file_does_not_exist.json", validate=True)

    with pytest.raises(gen.InvalidConfigData):
        data_reader.read_value_from_file(None, validate=False)

    test_dict_1 = {"Value": 5.0}
    with open(tmp_test_directory / "test_read_value_from_file_1.json", "w", encoding="utf-8") as f:
        json.dump(test_dict_1, f)
    assert isinstance(
        data_reader.read_value_from_file(
            tmp_test_directory / "test_read_value_from_file_1.json", validate=False
        ),
        float,
    )

    test_dict_2 = {"Value": "string_test"}
    with open(tmp_test_directory / "test_read_value_from_file_2.json", "w", encoding="utf-8") as f:
        json.dump(test_dict_2, f)
    assert (
        data_reader.read_value_from_file(
            tmp_test_directory / "test_read_value_from_file_2.json", validate=False
        )
        == "string_test"
    )

    test_dict_3 = {"No_Value": "no_value"}
    with open(tmp_test_directory / "test_read_value_from_file_3.json", "w", encoding="utf-8") as f:
        json.dump(test_dict_3, f)
    assert (
        data_reader.read_value_from_file(
            tmp_test_directory / "test_read_value_from_file_3.json", validate=False
        )
        is None
    )


def test_read_value_from_file_and_validate(caplog, tmp_test_directory):
    # schema file from metadata in file
    with caplog.at_level(logging.DEBUG):
        data_reader.read_value_from_file(
            "tests/resources/reference_point_altitude.json", validate=True
        )
        assert "Successful validation of yaml/json file" in caplog.text

    # schema explicitly given
    schema_file = (
        "https://raw.githubusercontent.com/gammasim/simulation_model/"
        "main/schema/reference_point_altitude.schema.yml"
    )
    with caplog.at_level(logging.DEBUG):
        data_reader.read_value_from_file(
            "tests/resources/reference_point_altitude.json",
            schema_file=schema_file,
            validate=True,
        )
        assert "Successful validation of yaml/json file" in caplog.text

    # no schema given
    test_dict_1 = {"Value": 5.0}
    with open(tmp_test_directory / "test_read_value_from_file_1.json", "w", encoding="utf-8") as f:
        json.dump(test_dict_1, f)
    with pytest.raises(jsonschema.exceptions.ValidationError):
        data_reader.read_value_from_file(
            tmp_test_directory / "test_read_value_from_file_1.json", validate=True
        ),

    # schema_file in meta_schema_url
    with caplog.at_level(logging.DEBUG):
        data_reader.read_value_from_file(
            "tests/resources/MST_mirror_2f_measurements.schema.yml", validate=True
        )
        assert "Successful validation of yaml/json file" in caplog.text
