#!/usr/bin/python3

import logging
from pathlib import Path

import pytest
from astropy.table import Table

from simtools.simtel.simtel_histograms import SimtelHistograms
from simtools.utils.general import read_hdf5

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
    return SimtelHistograms(histogram_files=simtel_array_histograms_file, test=True)


def test_histograms(io_handler, simtel_array_histograms_file):
    histogram_files = list()
    histogram_files.append(simtel_array_histograms_file)
    histogram_files.append(
        io_handler.get_input_data_file(
            file_name="run2_gamma_za20deg_azm0deg-North-Prod5_test-production-5.hdata.zst",
            test=True,
        )
    )

    hists = SimtelHistograms(histogram_files=histogram_files, test=True)

    fig_name = io_handler.get_output_file(
        file_name="simtel_histograms.pdf", sub_dir="plots", dir_type="test"
    )
    hists.plot_and_save_figures(fig_name=fig_name)

    assert fig_name.exists()


def test_meta_dict(simtel_array_histograms_instance):
    assert "simtools_version" in simtel_array_histograms_instance._meta_dict


def test_export_histograms(simtel_array_histograms_instance, io_handler):
    # Default values
    simtel_array_histograms_instance.export_histograms("test_hist_simtel.hdf5")

    file_name = Path(simtel_array_histograms_instance.output_path).joinpath("test_hist_simtel.hdf5")
    assert io_handler.get_output_directory(dir_type="test").joinpath(file_name).exists()

    # Read hdf5 file
    list_of_tables = read_hdf5(io_handler.get_output_directory(dir_type="test").joinpath(file_name))
    assert len(list_of_tables) == 10
    for table in list_of_tables:
        assert isinstance(table, Table)
    # Check piece of metadata
    print(list_of_tables[-1].meta)
    assert (
        list_of_tables[-1].meta["corsika_version"]
        == simtel_array_histograms_instance.corsika_version
    )
