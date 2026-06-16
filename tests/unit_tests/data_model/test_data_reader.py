#!/usr/bin/python3

import logging

import pytest
from astropy.io.registry.base import IORegistryError
from astropy.table import Table

from simtools.constants import TEST_RESOURCES_STATIC
from simtools.data_model import data_reader

logger = logging.getLogger()


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
