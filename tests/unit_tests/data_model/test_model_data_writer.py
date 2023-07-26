#!/usr/bin/python3

import logging
import re
from pathlib import Path

import pytest

import astropy

import simtools.data_model.model_data_writer as writer

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_write_metadata(tmp_test_directory):

    # test writer of metadata
    _metadata = {'name': 'test_metadata'}
    w_1 = writer.ModelDataWriter()
    with pytest.raises(TypeError):
        w_1.write_metadata(metadata=_metadata)

    yml_file = w_1.write_metadata(
        metadata=_metadata,
        ymlfile=tmp_test_directory.join("test_file.yml")
    )
    assert Path(yml_file).exists()

    with pytest.raises(FileNotFoundError):
        w_1.write_metadata(
            metadata=_metadata,
            ymlfile="./this_directory_is_not_theta/test_file.yml"
        )


def test_write_data():

    w_2 = writer.ModelDataWriter()
    w_2.write_data(None)

    empty_table = astropy.table.Table()
    with pytest.raises(astropy.io.registry.base.IORegistryError):
        w_2.write_data(empty_table)

def test_astropy_data_format():

    assert writer.ModelDataWriter._astropy_data_format("hdf5") == "hdf5"
    assert writer.ModelDataWriter._astropy_data_format("ecsv") == "ascii.ecsv"
