#!/usr/bin/python3

import logging
from pathlib import Path
from unittest.mock import patch

import astropy.units as u
import pytest
from astropy.io.registry.base import IORegistryError
from astropy.table import Table

import simtools.data_model.metadata_collector as metadata_collector
import simtools.data_model.model_data_writer as writer
from simtools.constants import MODEL_PARAMETER_SCHEMA_PATH, SCHEMA_PATH
from simtools.data_model import schema
from simtools.io import ascii_handler
from simtools.utils import names

logger = logging.getLogger()


test_file_2 = "test_file_2.ecsv"
ascii_format = "ascii.ecsv"


@pytest.fixture
def num_gains_schema_file():
    return MODEL_PARAMETER_SCHEMA_PATH / "num_gains.schema.yml"


@pytest.fixture
def num_gains_schema(num_gains_schema_file):
    return ascii_handler.collect_data_from_file(file_name=num_gains_schema_file)


def test_write(tmp_test_directory, args_dict_site):
    # both none (no exception expected)
    w_1 = writer.ModelDataWriter(output_path=tmp_test_directory)
    assert w_1.write(metadata=None, product_data=None) is None

    # metadata not none; no data and metadata file
    _metadata = metadata_collector.MetadataCollector(args_dict=args_dict_site)
    w_1.product_data_file = tmp_test_directory.join("test_file.ecsv")
    metadata_file = tmp_test_directory.join("test_file.meta.yml")
    w_1.write(metadata=_metadata, product_data=None)
    assert not metadata_file.exists()
    assert not Path(w_1.product_data_file).exists()

    # product_data not none - expect data file to be written; no metadata file
    empty_table = Table()
    w_1.write(metadata=None, product_data=empty_table)
    assert Path(w_1.product_data_file).exists()
    assert not metadata_file.exists()

    # both not none
    data = {"pixel": [25, 30, 28]}
    small_table = Table(data)
    w_1.product_data_file = tmp_test_directory.join(test_file_2)
    w_1.write(metadata=_metadata, product_data=small_table)
    assert Path(w_1.product_data_file).exists()
    assert (
        (Path(tmp_test_directory) / test_file_2).with_suffix(".integration_test.meta.yml").exists()
    )

    # check that table and metadata is good
    table = Table.read(w_1.product_data_file, format=ascii_format)
    assert "pixel" in table.colnames
    assert "cta" in table.meta.keys()

    w_1.product_data_format = "not_an_astropy_format"
    with pytest.raises(IORegistryError):
        w_1.write(metadata=None, product_data=empty_table)

    # test json format
    dict_data = {"value": 5.5}
    w_1.product_data_file = tmp_test_directory.join("test_file.json")
    w_1.write(metadata=None, product_data=dict_data)
    assert Path(w_1.product_data_file).is_file()


def test_write_dict_to_model_parameter_json(tmp_test_directory):
    w1 = writer.ModelDataWriter(output_path=tmp_test_directory)
    data_dict = {"value": 5.5}
    data_file = tmp_test_directory.join("test_file.json")
    w1.write_dict_to_model_parameter_json(file_name=data_file, data_dict=data_dict)
    assert Path(data_file).is_file()


def test_dump(args_dict):
    empty_table = Table()

    args_dict["output_file"] = "test_file.ecsv"
    args_dict["skip_output_validation"] = True
    writer.ModelDataWriter().dump(
        args_dict=args_dict,
        metadata=None,
        product_data=empty_table,
        validate_schema_file=None,
    )

    assert Path(args_dict["output_path"]).joinpath(args_dict["output_file"]).exists()

    # Test only that output validation is queried, as the validation itself is
    # tested in test_validate_and_transform (therefore: expect KeyError)
    args_dict["skip_output_validation"] = False
    with pytest.raises(KeyError):
        writer.ModelDataWriter().dump(
            args_dict=args_dict,
            metadata=None,
            product_data=empty_table,
            validate_schema_file=SCHEMA_PATH / "input/MST_mirror_2f_measurements.schema.yml",
        )

    # explicitly set output_file
    writer.ModelDataWriter().dump(
        args_dict=args_dict,
        output_file=test_file_2,
        metadata=None,
        product_data=empty_table,
        validate_schema_file=None,
    )
    assert Path(args_dict["output_path"]).joinpath(test_file_2).exists()


def test_validate_and_transform(num_gains_schema_file):
    w_1 = writer.ModelDataWriter()
    with pytest.raises(TypeError):
        w_1.validate_and_transform(product_data_table=None, validate_schema_file=None)

    _table = Table.read("tests/resources/MLTdata-preproduction.ecsv", format=ascii_format)
    return_table = w_1.validate_and_transform(
        product_data_table=_table,
        validate_schema_file=SCHEMA_PATH / "input/MST_mirror_2f_measurements.schema.yml",
    )
    assert len(_table.columns) == len(return_table.columns)

    num_gains = {
        "parameter": "num_gains",
        "instrument": "LSTN-01",
        "site": "North",
        "version": "6.0.0",
        "value": 2,
        "unit": None,
        "type": "int",
        "applicable": True,
        "file": False,
    }

    return_dict = w_1.validate_and_transform(
        product_data_dict=num_gains,
        validate_schema_file=num_gains_schema_file,
    )
    assert isinstance(return_dict, dict)

    num_gains["value"] = 25
    with pytest.raises(ValueError, match=r"^Value for column '0' out of range."):
        w_1.validate_and_transform(
            product_data_dict=num_gains,
            validate_schema_file=num_gains_schema_file,
        )


def test_astropy_data_format():
    assert writer.ModelDataWriter._astropy_data_format("hdf5") == "hdf5"
    assert writer.ModelDataWriter._astropy_data_format("ecsv") == ascii_format
    assert writer.ModelDataWriter._astropy_data_format(ascii_format) == ascii_format


def test_dump_model_parameter(tmp_test_directory, db_config):
    parameter_version = "1.1.0"
    instrument = "LSTN-01"
    num_gains_name = "num_gains"
    # single value, no unit
    num_gains_dict = writer.ModelDataWriter.dump_model_parameter(
        parameter_name=num_gains_name,
        value=2,
        instrument=instrument,
        parameter_version=parameter_version,
        output_file="num_gains.json",
        output_path=tmp_test_directory,
    )
    assert (Path(tmp_test_directory) / "num_gains.json").is_file()
    assert isinstance(num_gains_dict, dict)
    assert num_gains_dict["value"] == 2
    assert num_gains_dict["unit"] == u.dimensionless_unscaled

    # list of value, with unit
    position_dict = writer.ModelDataWriter.dump_model_parameter(
        parameter_name="array_element_position_utm",
        value=[217.6596 * u.km, 3184.9951 * u.km, 218500.0 * u.cm],
        instrument=instrument,
        parameter_version=parameter_version,
        output_file="array_element_position_utm.json",
        output_path=tmp_test_directory,
        metadata_input_dict={"name": "test_metadata"},
    )
    assert (Path(tmp_test_directory) / "array_element_position_utm.json").is_file()
    assert isinstance(position_dict, dict)
    assert position_dict["value"][0] == pytest.approx(217659.6)
    assert position_dict["value"][1] == pytest.approx(3184995.1)
    assert position_dict["value"][2] == pytest.approx(2185.0)
    assert Path(tmp_test_directory / "array_element_position_utm.meta.yml").is_file()

    position_dict = writer.ModelDataWriter.dump_model_parameter(
        parameter_name="focus_offset",
        value=[6.55 * u.cm, 0.0 * u.deg, 0.0, 0.0],
        instrument="LSTN-01",
        parameter_version=parameter_version,
        output_file="focus_offset.json",
        output_path=tmp_test_directory,
    )
    assert position_dict["value"][0] == pytest.approx(6.55)
    assert position_dict["value"][1] == pytest.approx(0.0)
    assert position_dict["value"][2] == pytest.approx(0.0)
    assert position_dict["value"][3] == pytest.approx(0.0)

    with patch(
        "simtools.data_model.model_data_writer.ModelDataWriter.check_db_for_existing_parameter"
    ) as mock_db_check:
        writer.ModelDataWriter.dump_model_parameter(
            parameter_name=num_gains_name,
            value=2,
            instrument=instrument,
            parameter_version=parameter_version,
            output_file=num_gains_name + ".json",
            output_path=tmp_test_directory,
            db_config=db_config,
        )
        mock_db_check.assert_called_once_with(
            num_gains_name, instrument, parameter_version, db_config
        )


def test_get_validated_parameter_dict():
    w1 = writer.ModelDataWriter()
    assert w1.get_validated_parameter_dict(
        parameter_name="num_gains", value=2, instrument="MSTN-01", parameter_version="0.0.1"
    ) == {
        "schema_version": schema.get_model_parameter_schema_version(),
        "parameter": "num_gains",
        "instrument": "MSTN-01",
        "site": "North",
        "parameter_version": "0.0.1",
        "unique_id": None,
        "value": 2,
        "unit": u.Unit(""),
        "type": "int64",
        "file": False,
        "meta_parameter": False,
        "model_parameter_schema_version": "0.1.0",
    }

    assert w1.get_validated_parameter_dict(
        parameter_name="transit_time_error",
        value=5.0 * u.ns,
        instrument="LSTN-01",
        parameter_version="0.0.1",
    ) == {
        "schema_version": schema.get_model_parameter_schema_version(),
        "parameter": "transit_time_error",
        "instrument": "LSTN-01",
        "site": "North",
        "parameter_version": "0.0.1",
        "unique_id": None,
        "value": 5,
        "unit": u.Unit("ns"),
        "type": "float64",
        "file": False,
        "meta_parameter": False,
        "model_parameter_schema_version": "0.1.0",
    }

    assert w1.get_validated_parameter_dict(
        parameter_name="reference_point_altitude",
        value=2.7 * u.km,
        instrument="North",
        parameter_version="0.0.1",
    ) == {
        "schema_version": schema.get_model_parameter_schema_version(),
        "parameter": "reference_point_altitude",
        "instrument": "North",
        "site": "North",
        "parameter_version": "0.0.1",
        "unique_id": None,
        "value": 2700.0,
        "unit": u.Unit("m"),
        "type": "float64",
        "file": False,
        "meta_parameter": False,
        "model_parameter_schema_version": "0.1.0",
    }


def test_prepare_data_dict_for_writing():
    data_dict_5 = {
        "value": [5.5, 6.6],
        "unit": ["None", "None"],
        "type": "float64",
    }
    assert writer.ModelDataWriter.prepare_data_dict_for_writing(data_dict_5) == {
        "value": [5.5, 6.6],
        "unit": "null",
        "type": "float64",
    }
    data_dict_6 = {
        "value": 5.0,
        "unit": "None",
        "type": "float64",
    }
    assert writer.ModelDataWriter.prepare_data_dict_for_writing(data_dict_6) == {
        "value": 5.0,
        "unit": "null",
        "type": "float64",
    }


def test_get_unit_from_schema(num_gains_schema):
    w1 = writer.ModelDataWriter()

    assert w1._get_unit_from_schema() is None

    w1.schema_dict = num_gains_schema

    w1.schema_dict["data"][0]["unit"] = "m"
    assert w1._get_unit_from_schema() == "m"

    w1.schema_dict["data"][0]["unit"] = "dimensionless"
    assert w1._get_unit_from_schema() is None

    w1.schema_dict["data"][0].pop("unit")
    assert w1._get_unit_from_schema() is None


def test_parameter_is_a_file(num_gains_schema):
    w1 = writer.ModelDataWriter()

    assert not w1._parameter_is_a_file()

    w1.schema_dict = num_gains_schema

    w1.schema_dict["data"][0]["type"] = "file"
    assert w1._parameter_is_a_file()

    w1.schema_dict["data"][0].pop("type")
    assert not w1._parameter_is_a_file()

    w1.schema_dict["data"] = []
    assert not w1._parameter_is_a_file()


def test_check_db_for_existing_parameter():
    db_config = {"host": "localhost", "port": 27017}
    parameter_name = "test_parameter"
    instrument = "LSTN-01"
    parameter_version = "1.0.0"

    w1 = writer.ModelDataWriter()

    with patch("simtools.data_model.model_data_writer.db_handler.DatabaseHandler") as mockdbhandler:
        mock_db_instance = mockdbhandler.return_value
        mock_db_instance.get_model_parameter.side_effect = ValueError("Parameter not found")

        # Test case where parameter does not exist
        w1.check_db_for_existing_parameter(parameter_name, instrument, parameter_version, db_config)
        mock_db_instance.get_model_parameter.assert_called_once_with(
            parameter=parameter_name,
            parameter_version=parameter_version,
            site=names.get_site_from_array_element_name(instrument),
            array_element_name=instrument,
        )

        # Reset mock for next test
        mock_db_instance.get_model_parameter.reset_mock()

        # Test case where parameter exists
        mock_db_instance.get_model_parameter.side_effect = None
        with pytest.raises(
            ValueError,
            match=f"Parameter {parameter_name} with version {parameter_version} already exists.",
        ):
            w1.check_db_for_existing_parameter(
                parameter_name, instrument, parameter_version, db_config
            )
        mock_db_instance.get_model_parameter.assert_called_once_with(
            parameter=parameter_name,
            parameter_version=parameter_version,
            site=names.get_site_from_array_element_name(instrument),
            array_element_name=instrument,
        )


def test_find_highest_schema_version():
    w1 = writer.ModelDataWriter()

    # Test with single entry
    schema_list_single = [{"schema_version": "1.0.0", "data": []}]
    result = w1._find_highest_schema_version(schema_list_single)
    assert result["schema_version"] == "1.0.0"

    # Test with multiple entries - semantic versioning
    schema_list_multiple = [
        {"schema_version": "1.0.0", "data": []},
        {"schema_version": "2.1.0", "data": []},
        {"schema_version": "1.5.0", "data": []},
    ]
    result = w1._find_highest_schema_version(schema_list_multiple)
    assert result["schema_version"] == "2.1.0"

    # Test with patch versions
    schema_list_patches = [
        {"schema_version": "1.0.1", "data": []},
        {"schema_version": "1.0.0", "data": []},
        {"schema_version": "1.0.2", "data": []},
    ]
    result = w1._find_highest_schema_version(schema_list_patches)
    assert result["schema_version"] == "1.0.2"

    # Test with mixed entries (some without schema_version)
    schema_list_mixed = [
        {"data": []},
        {"schema_version": "1.0.0", "data": []},
        {"other_field": "value"},
        {"schema_version": "2.0.0", "data": []},
    ]
    result = w1._find_highest_schema_version(schema_list_mixed)
    assert result["schema_version"] == "2.0.0"

    with pytest.raises(TypeError, match=r"No valid schema versions found in the list\."):
        w1._find_highest_schema_version(None)


def test_read_schema_dict(tmp_test_directory):
    w1 = writer.ModelDataWriter()

    # Test with non-existent parameter
    with pytest.raises(FileNotFoundError):
        w1._read_schema_dict("non_existent_parameter", None)

    # Test with existing parameter (num_gains)
    schema_dict, schema_file = w1._read_schema_dict("num_gains", None)
    assert isinstance(schema_dict, dict)
    assert "data" in schema_dict
    assert schema_file.name == "num_gains.schema.yml"

    # Test with specific schema version (mock list scenario)
    with (
        patch("simtools.io.ascii_handler.collect_data_from_file") as mock_collect,
        patch("simtools.data_model.schema.get_model_parameter_schema_file") as mock_schema_file,
    ):
        mock_collect.return_value = [
            {"schema_version": "1.0.0", "data": []},
            {"schema_version": "2.0.0", "data": []},
        ]
        mock_schema_file.return_value = Path("test_param.schema.yml")
        schema_dict, schema_file = w1._read_schema_dict("test_param", "1.0.0")
        assert schema_dict["schema_version"] == "1.0.0"

    # Test with list and no schema version specified (should get highest version)
    with (
        patch("simtools.io.ascii_handler.collect_data_from_file") as mock_collect,
        patch("simtools.data_model.schema.get_model_parameter_schema_file") as mock_schema_file,
    ):
        mock_collect.return_value = [
            {"schema_version": "1.0.0", "data": []},
            {"schema_version": "2.0.0", "data": []},
            {"schema_version": "1.5.0", "data": []},
        ]
        mock_schema_file.return_value = Path("test_param.schema.yml")
        schema_dict, schema_file = w1._read_schema_dict("test_param", None)
        assert schema_dict["schema_version"] == "2.0.0"

    # Test with list but no matching version
    with (
        patch("simtools.io.ascii_handler.collect_data_from_file") as mock_collect,
        patch("simtools.data_model.schema.get_model_parameter_schema_file") as mock_schema_file,
    ):
        mock_collect.return_value = [
            {"schema_version": "1.0.0", "data": []},
            {"schema_version": "2.0.0", "data": []},
        ]
        mock_schema_file.return_value = Path("test_param.schema.yml")
        with pytest.raises(ValueError, match=r"Schema version 3\.0\.0 not found for test_param"):
            w1._read_schema_dict("test_param", "3.0.0")
