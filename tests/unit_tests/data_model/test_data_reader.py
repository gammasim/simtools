#!/usr/bin/python3

import json
import logging

import astropy.units as u
import pytest
from astropy.io.registry.base import IORegistryError
from astropy.table import Table

from simtools.constants import TEST_RESOURCES_STATIC
from simtools.data_model import data_reader, schema

logger = logging.getLogger()

JSON_TEST_FILE = "test_read_value_from_file_1.json"


def test_read_table_from_file(get_test_data_file):
    assert isinstance(
        data_reader.read_table_from_file(get_test_data_file("telescope_positions", "North")),
        Table,
    )

    with pytest.raises(FileNotFoundError):
        data_reader.read_table_from_file("non_existing_file.fits")

    with pytest.raises(IORegistryError):
        data_reader.read_table_from_file(None)


def test_read_table_from_file_and_validate(get_test_data_file):
    # schema file from metadata in table
    assert isinstance(
        data_reader.read_table_from_file(
            get_test_data_file("telescope_positions", "North"), validate=True
        ),
        Table,
    )

    assert isinstance(
        data_reader.read_table_from_file(
            f"{TEST_RESOURCES_STATIC}/telescope_positions-North-utm-without-cta-meta.ecsv",
            validate=True,
            metadata_file=f"{TEST_RESOURCES_STATIC}/telescope_positions-North-utm.meta.yml",
        ),
        Table,
    )


def test_read_value_from_file(tmp_test_directory, model_parameter_json):
    assert isinstance(
        data_reader.read_value_from_file(model_parameter_json),
        u.quantity.Quantity,
    )

    with pytest.raises(FileNotFoundError):
        data_reader.read_value_from_file("this_file_does_not_exist.json", validate=True)

    with pytest.raises(TypeError):
        data_reader.read_value_from_file(None, validate=False)

    test_dict_1 = {"value": 5.0}
    with open(tmp_test_directory / JSON_TEST_FILE, "w", encoding="utf-8") as f:
        json.dump(test_dict_1, f)
    assert isinstance(
        data_reader.read_value_from_file(tmp_test_directory / JSON_TEST_FILE, validate=False),
        float,
    )

    test_dict_2 = {"value": "string_test"}
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


def test_read_value_from_file_and_validate(caplog, model_parameter_json):
    # schema explicitly given
    with caplog.at_level("DEBUG"):
        validated_value = data_reader.read_value_from_file(
            model_parameter_json,
            schema_file=schema.get_model_parameter_schema_file("array_element_position_ground"),
            validate=True,
        )
    assert "Successful validation of yaml/json file" in caplog.text
    assert isinstance(validated_value, u.quantity.Quantity)
    assert validated_value.unit == u.Unit("m")


def test_read_value_from_file_and_validate_uses_metadata_schema(monkeypatch, model_parameter_json):
    class DummyMetadataCollector:
        def __init__(self, *args, **kwargs):
            pass

        def get_data_model_schema_file_name(self):
            return schema.get_model_parameter_schema_file("array_element_position_ground")

    monkeypatch.setattr(data_reader, "MetadataCollector", DummyMetadataCollector)

    validated_value = data_reader.read_value_from_file(model_parameter_json, validate=True)

    assert isinstance(validated_value, u.quantity.Quantity)
    assert validated_value.unit == u.Unit("m")


def test_collapse_unit_handles_scalar_and_heterogeneous_units():
    assert data_reader._collapse_unit(1) == 1

    with pytest.raises(ValueError, match="Cannot collapse heterogeneous units"):
        data_reader._collapse_unit(["m", "s"])
