#!/usr/bin/python3

import logging

import pytest
from astropy.table import Table

from simtools.io_operations.hdf5_handler import read_hdf5
from simtools.simtel.simtel_histograms import SimtelHistograms

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def simtel_array_histograms_file(io_handler, corsika_output_file_name):
    return io_handler.get_input_data_file(
        file_name="run1_gamma_za20deg_azm0deg-North-Prod5_test-production-5.hdata.zst",
        test=True,
    )


@pytest.fixture
def simtel_array_histograms_instance(simtel_array_histograms_file):
    instance = SimtelHistograms(histogram_files=simtel_array_histograms_file, test=True)
    instance.combine_histogram_files()
    return instance


def test_meta_dict(simtel_array_histograms_instance):
    assert "simtools_version" in simtel_array_histograms_instance._meta_dict


def test_export_histograms(simtel_array_histograms_instance, io_handler):
    file_with_path = io_handler.get_output_directory(dir_type="test").joinpath(
        "test_hist_simtel.hdf5"
    )
    # Default values
    simtel_array_histograms_instance.export_histograms(file_with_path)

    assert file_with_path.exists()

    # Read hdf5 file
    list_of_tables = read_hdf5(file_with_path)
    assert len(list_of_tables) == 144
    for table in list_of_tables:
        assert isinstance(table, Table)
