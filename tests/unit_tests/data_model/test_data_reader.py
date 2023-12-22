#!/usr/bin/python3

import pytest
from astropy.io.registry.base import IORegistryError
from astropy.table import Table

from simtools.data_model import data_reader


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
            "tests/resources/telescope_positions-North-utm-withoutmeta.ecsv",
            validate=True,
            metadata_file="tests/resources/telescope_positions-North-utm.meta.yml",
        ),
        Table,
    )
