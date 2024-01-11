#!/usr/bin/python3

import logging
from pathlib import Path

import astropy
import pytest

import simtools.data_model.model_data_writer as writer

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_write(tmp_test_directory):
    # both none (no exception expected)
    w_1 = writer.ModelDataWriter()
    w_1.write(metadata=None, product_data=None)

    # metadata not none
    _metadata = {"name": "test_metadata"}
    w_1.product_data_file = tmp_test_directory.join("test_file.ecsv")
    w_1.write(metadata=_metadata, product_data=None)

    # product_data not none
    empty_table = astropy.table.Table()
    w_1.write(metadata=None, product_data=empty_table)

    assert Path(w_1.product_data_file).exists()

    # both not none
    data = {"pixel": [25, 30, 28]}
    small_table = astropy.table.Table(data)
    w_1.product_data_file = tmp_test_directory.join("test_file_2.ecsv")
    w_1.write(metadata=_metadata, product_data=small_table)
    assert Path(w_1.product_data_file).exists()

    # check that table and metadata is good
    table = astropy.table.Table.read(w_1.product_data_file, format="ascii.ecsv")
    assert "pixel" in table.colnames
    assert "NAME" in table.meta.keys()

    w_1.product_data_format = "not_an_astropy_format"
    with pytest.raises(astropy.io.registry.base.IORegistryError):
        w_1.write(metadata=None, product_data=empty_table)


def test_dump(args_dict, tmp_test_directory):
    _metadata = {"name": "test_metadata"}
    empty_table = astropy.table.Table()

    args_dict["use_plain_output_path"] = True
    args_dict["output_file"] = "test_file.ecsv"
    args_dict["skip_output_validation"] = True
    writer.ModelDataWriter().dump(
        args_dict=args_dict,
        metadata=_metadata,
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
            metadata=_metadata,
            product_data=empty_table,
            validate_schema_file="tests/resources/MST_mirror_2f_measurements.schema.yml",
        )

    # explicitly set output_file
    writer.ModelDataWriter().dump(
        args_dict=args_dict,
        output_file="test_file_2.ecsv",
        metadata=_metadata,
        product_data=empty_table,
        validate_schema_file=None,
    )
    assert Path(args_dict["output_path"]).joinpath("test_file_2.ecsv").exists()


def test_validate_and_transform(tmp_test_directory):
    w_1 = writer.ModelDataWriter()
    with pytest.raises(TypeError):
        w_1.validate_and_transform(product_data=None, validate_schema_file=None)

    _table = astropy.table.Table.read(
        "tests/resources/MLTdata-preproduction.ecsv", format="ascii.ecsv"
    )
    return_table = w_1.validate_and_transform(
        product_data=_table,
        validate_schema_file="tests/resources/MST_mirror_2f_measurements.schema.yml",
    )
    assert len(_table.columns) == len(return_table.columns)


def test_write_metadata_to_yml(tmp_test_directory):
    # test writer of metadata
    _metadata = {"name": "test_metadata"}
    w_1 = writer.ModelDataWriter()
    with pytest.raises(TypeError):
        w_1.write_metadata_to_yml(metadata=_metadata)

    yml_file = w_1.write_metadata_to_yml(
        metadata=_metadata, yml_file=tmp_test_directory.join("test_file.yml")
    )
    assert Path(yml_file).exists()

    with pytest.raises(FileNotFoundError):
        w_1.write_metadata_to_yml(
            metadata=_metadata, yml_file="./this_directory_is_not_there/test_file.yml"
        )

    with pytest.raises(AttributeError):
        w_1.write_metadata_to_yml(metadata=None, yml_file=tmp_test_directory.join("test_file.yml"))

    with pytest.raises(TypeError):
        w_1.write_metadata_to_yml(
            metadata=_metadata,
            yml_file=None,
        )


def test_astropy_data_format():
    assert writer.ModelDataWriter._astropy_data_format("hdf5") == "hdf5"
    assert writer.ModelDataWriter._astropy_data_format("ecsv") == "ascii.ecsv"
    assert writer.ModelDataWriter._astropy_data_format("ascii.ecsv") == "ascii.ecsv"
