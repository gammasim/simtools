#!/usr/bin/python3

import json
import logging
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

import astropy.units as u
import pytest
from astropy.io.registry.base import IORegistryError
from astropy.table import Table

import simtools.data_model.metadata_collector as metadata_collector
import simtools.data_model.model_data_writer as writer
from simtools.constants import MODEL_PARAMETER_SCHEMA_PATH, SCHEMA_PATH
from simtools.data_model.product_checksum import (
    embedded_sha256,
    sha256_of_file,
    verify_sha256,
)
from simtools.io import ascii_handler
from simtools.io.output_paths import OutputPaths
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
    result = w_1.write_data(metadata=None, product_data=None)
    assert result is None

    # metadata not none; no data and metadata file
    _metadata = metadata_collector.MetadataCollector(args_dict=args_dict_site)
    w_1.output_file = tmp_test_directory.join("test_file.ecsv")
    metadata_file = tmp_test_directory.join("test_file.meta.yml")
    w_1.write_data(metadata=_metadata, product_data=None)
    assert not metadata_file.exists()
    assert not Path(w_1.output_file).exists()

    # product_data not none - expect data file to be written; no metadata file
    empty_table = Table()
    w_1.write_data(metadata=None, product_data=empty_table)
    assert Path(w_1.output_file).exists()
    assert not metadata_file.exists()

    # both not none
    data = {"pixel": [25, 30, 28]}
    small_table = Table(data)
    w_1.output_file = tmp_test_directory.join(test_file_2)
    w_1.write_data(metadata=_metadata, product_data=small_table)
    assert Path(w_1.output_file).exists()
    assert (
        (Path(tmp_test_directory) / test_file_2).with_suffix(".integration_test.meta.yml").exists()
    )

    # check that table and metadata is good
    table = Table.read(w_1.output_file, format=ascii_format)
    assert "pixel" in table.colnames
    assert "cta" in table.meta.keys()

    w_1.output_file_format = "not_an_astropy_format"
    with pytest.raises(IORegistryError):
        w_1.write_data(metadata=None, product_data=empty_table)

    # test json format
    dict_data = {"value": 5.5}
    w_1.output_file = tmp_test_directory.join("test_file.json")
    w_1.write_data(metadata=None, product_data=dict_data)
    assert Path(w_1.output_file).is_file()


def test_dump(args_dict, run_context):
    context = run_context.replace(
        args=args_dict,
        paths=OutputPaths(args_dict["output_path"], args_dict.get("model_path")),
    )
    empty_table = Table()

    output_file = "test_file.ecsv"
    writer.ModelDataWriter().write_product_data(
        output_file=output_file,
        metadata=None,
        product_data=empty_table,
        validate_schema_file=None,
        context=context,
    )

    assert Path(args_dict["output_path"]).joinpath(output_file).exists()

    # Test only that output validation is queried, as the validation itself is
    # tested in test_validate_and_transform (therefore: expect KeyError)
    args_dict["skip_output_validation"] = False
    context = context.replace(args=args_dict)
    with pytest.raises(KeyError):
        writer.ModelDataWriter().write_product_data(
            output_file=output_file,
            metadata=None,
            product_data=empty_table,
            validate_schema_file=SCHEMA_PATH / "input/MST_mirror_2f_measurements.schema.yml",
            context=context,
        )


def test_validate_and_transform(num_gains_schema_file):
    w_1 = writer.ModelDataWriter()
    with pytest.raises(TypeError):
        w_1.validate_and_transform(product_data_table=None, validate_schema_file=None)

    _table = Table(
        {
            "mirror_panel_id": ["1"],
            "mirror_curvature_radius": [100.0] * u.cm,
            "psf": [1.0] * u.cm,
            "psf_opt": [0.8] * u.cm,
        }
    )
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


def test_derive_data_format():
    assert writer.ModelDataWriter._derive_data_format("hdf5") == "hdf5"
    assert writer.ModelDataWriter._derive_data_format("ecsv") == ascii_format
    assert writer.ModelDataWriter._derive_data_format(ascii_format) == ascii_format
    assert writer.ModelDataWriter._derive_data_format(None) is None
    assert writer.ModelDataWriter._derive_data_format(None, output_file="file.ecsv") == ascii_format
    assert writer.ModelDataWriter._derive_data_format(None, output_file="file.hdf5") == "hdf5"


def test_write_model_parameter_does_not_write_metadata_on_validation_failure(tmp_test_directory):
    output_file = "num_gains.json"

    with pytest.raises(ValueError, match=r"^Value for column '0' out of range."):
        writer.ModelDataWriter.write_model_parameter(
            parameter_name="num_gains",
            value=25,
            instrument="LSTN-01",
            parameter_version="1.1.0",
            output_file=output_file,
            output_path=tmp_test_directory,
            metadata_input_dict={"name": "test_metadata"},
            check_for_existing_parameter=False,
        )

    assert not (Path(tmp_test_directory) / output_file).exists()
    assert not Path(tmp_test_directory / "num_gains.meta.yml").exists()


def test_get_parameter_type_for_schema_uses_selected_schema_version():
    writer_instance = writer.ModelDataWriter()

    assert writer_instance.get_parameter_type_for_schema("fadc_pulse_shape", "0.2.0") == "dict"
    assert writer_instance.get_parameter_type_for_schema("fadc_pulse_shape", "0.1.0") == "file"


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

    # empty list unit (produced when a dimensionless float goes through the pipeline) -> None
    data_dict_7 = {"value": 0.784, "unit": [], "type": "float64"}
    assert writer.ModelDataWriter.prepare_data_dict_for_writing(data_dict_7) == {
        "value": 0.784,
        "unit": None,
        "type": "float64",
    }


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


def test_check_model_reader_for_existing_parameter():
    parameter_name = "test_parameter"
    instrument = "LSTN-01"
    parameter_version = "1.0.0"

    model_reader = Mock()
    model_reader.is_configured.return_value = True
    model_reader.get_model_parameter.side_effect = ValueError("Parameter not found")
    w1 = writer.ModelDataWriter(model_reader=model_reader)

    # Test case where parameter does not exist
    w1.check_for_existing_parameter(parameter_name, instrument, parameter_version)
    model_reader.get_model_parameter.assert_called_once_with(
        parameter=parameter_name,
        parameter_version=parameter_version,
        site=names.get_site_from_array_element_name(instrument),
        array_element_name=instrument,
    )

    # Reset mock for next test
    model_reader.get_model_parameter.reset_mock()

    # Test case where parameter exists
    model_reader.get_model_parameter.side_effect = None
    with pytest.raises(
        ValueError,
        match=f"Parameter {parameter_name} with version {parameter_version} already exists.",
    ):
        w1.check_for_existing_parameter(parameter_name, instrument, parameter_version)
    model_reader.get_model_parameter.assert_called_once_with(
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


def test_write_model_parameter_without_metadata_has_unique_id(tmp_test_directory):
    parameter = writer.ModelDataWriter.write_model_parameter(
        parameter_name="num_gains",
        value=1,
        instrument="LSTN-01",
        parameter_version="1.1.0",
        output_file="num_gains.json",
        output_path=tmp_test_directory,
        check_for_existing_parameter=False,
    )
    assert parameter["unique_id"] is not None
    assert uuid.UUID(parameter["unique_id"]).version == 7
    written = json.loads((Path(tmp_test_directory) / "num_gains.json").read_text(encoding="utf-8"))
    assert written["unique_id"] == parameter["unique_id"]


def test_write_model_parameter_dict_json_fills_unique_id(tmp_test_directory):
    model_writer = writer.ModelDataWriter(output_path=tmp_test_directory)
    model_writer.write_model_parameter_dict_json(
        "position.json", {"parameter": "array_element_position_ground", "unique_id": None}
    )
    written = json.loads((Path(tmp_test_directory) / "position.json").read_text(encoding="utf-8"))
    assert uuid.UUID(written["unique_id"]).version == 7


def test_write_model_parameter_metadata_describes_parameter_schema(tmp_test_directory):
    writer.ModelDataWriter.write_model_parameter(
        parameter_name="altitude",
        value=2100.0,
        unit="m",
        instrument="OBS-North",
        parameter_version="9.9.9",
        output_file="altitude-9.9.9.json",
        output_path=tmp_test_directory,
        metadata_input_dict={"label": "test", "site": "North"},
        check_for_existing_parameter=False,
    )
    metadata_file = next(Path(tmp_test_directory).glob("altitude-9.9.9*.meta.yml"))
    product = ascii_handler.collect_data_from_file(metadata_file)["cta"]["product"]
    assert product["data"]["model"]["name"] == "altitude"
    assert product["data"]["model"]["version"] == "0.1.0"
    assert product["filename"] == "altitude-9.9.9.json"


def test_write_data_ecsv_without_metadata_embeds_cta_block_and_checksum(tmp_test_directory):
    model_writer = writer.ModelDataWriter(output_path=tmp_test_directory)
    model_writer.output_file = Path(tmp_test_directory) / "table_no_meta.ecsv"
    model_writer.write_data(product_data=Table({"pixel": [25, 30, 28]}))

    table = Table.read(model_writer.output_file, format=ascii_format)
    product = table.meta["cta"]["product"]
    assert product["format"] == "ecsv"
    assert product["sha256"] == embedded_sha256(model_writer.output_file)
    assert verify_sha256(model_writer.output_file)
    assert not (Path(tmp_test_directory) / "table_no_meta.meta.yml").exists()


def test_write_data_json_without_metadata_writes_sidecar_with_checksum(tmp_test_directory):
    model_writer = writer.ModelDataWriter(output_path=tmp_test_directory)
    model_writer.output_file = Path(tmp_test_directory) / "dict_no_meta.json"
    model_writer.write_data(product_data={"value": 5.5})

    sidecar = next(Path(tmp_test_directory).glob("dict_no_meta*.meta.yml"))
    product = ascii_handler.collect_data_from_file(sidecar)["cta"]["product"]
    assert product["sha256"] == sha256_of_file(model_writer.output_file)
    assert verify_sha256(model_writer.output_file, expected=product["sha256"])


def test_write_model_parameter_without_metadata_writes_cta_sidecar(tmp_test_directory):
    writer.ModelDataWriter.write_model_parameter(
        parameter_name="num_gains",
        value=1,
        instrument="LSTN-01",
        parameter_version="1.1.0",
        output_file="num_gains.json",
        output_path=tmp_test_directory,
        check_for_existing_parameter=False,
    )
    sidecar = next(Path(tmp_test_directory).glob("num_gains*.meta.yml"))
    cta = ascii_handler.collect_data_from_file(sidecar)["cta"]
    assert cta["product"]["id"] is not None
    assert cta["product"]["sha256"] == sha256_of_file(Path(tmp_test_directory) / "num_gains.json")
    assert cta["instrument"]["id"] == "LSTN-01"


def test_with_run_context_metadata_fills_only_missing_contact_and_application_entries():
    run_args = {
        "user_name": "run user",
        "user_email": "user@example.org",
        "user_orcid": None,
        "_application_description": "Submit a model parameter.",
        "output_path": "/local/output",
    }

    filled = writer._with_run_context_metadata({"user_name": "given"}, run_args)

    assert filled == {
        "user_name": "given",
        "user_email": "user@example.org",
        "_application_description": "Submit a model parameter.",
    }
