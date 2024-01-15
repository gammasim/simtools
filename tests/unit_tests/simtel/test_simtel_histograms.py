#!/usr/bin/python3

import logging

import matplotlib.pyplot as plt
import pytest
from astropy import units as u
from astropy.table import Table
from matplotlib.collections import QuadMesh

from simtools.io_operations.hdf5_handler import read_hdf5
from simtools.simtel.simtel_histograms import (
    InconsistentHistogramFormat,
    SimtelHistograms,
)

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


def test_number_of_histograms(simtel_array_histograms_file, simtel_array_histograms_instance):
    instance_alone = SimtelHistograms(histogram_files=simtel_array_histograms_file, test=True)
    assert instance_alone.number_of_histograms == 145
    assert (
        len(simtel_array_histograms_instance.combined_hists)
        == simtel_array_histograms_instance.number_of_histograms
    )


def test_get_histogram_title(simtel_array_histograms_file, simtel_array_histograms_instance):
    instance_alone = SimtelHistograms(histogram_files=simtel_array_histograms_file, test=True)
    assert instance_alone.number_of_histograms == 145
    for instance in [instance_alone, simtel_array_histograms_instance]:
        assert instance.get_histogram_title(0) == "Events, without weights (Ra, log10(E))"


def test_combine_histogram_files(simtel_array_histograms_file, caplog):
    # Reading one histogram file
    instance_alone = SimtelHistograms(histogram_files=simtel_array_histograms_file, test=True)

    # Passing the same file twice
    instance_all = SimtelHistograms(
        histogram_files=[simtel_array_histograms_file, simtel_array_histograms_file], test=True
    )

    assert (
        2 * instance_alone.combined_hists[0]["data"] == instance_all.combined_hists[0]["data"]
    ).all()

    # Test inconsistency
    instance_all.combined_hists[0]["lower_x"] = instance_all.combined_hists[0]["lower_x"] + 1
    with pytest.raises(InconsistentHistogramFormat):
        instance_all._combined_hists = None
        assert instance_all._combined_hists is None
        _ = instance_all.combined_hists
        assert "Trying to add histograms with inconsistent dimensions" in caplog.text


def test_plot_one_histogram(simtel_array_histograms_instance):
    fig, ax = plt.subplots()
    simtel_array_histograms_instance.plot_one_histogram(0, ax)
    quadmesh = ax.collections[0]
    assert isinstance(quadmesh, QuadMesh)


def test_trigger_rate_per_histogram(simtel_array_histograms_instance):
    trigger_rate = simtel_array_histograms_instance.trigger_rate_per_histogram(livetime=5 * u.h)
    assert pytest.approx(trigger_rate[0].value, 0.1) == 37972.1
    assert trigger_rate[0].unit == 1 / u.s
    trigger_rate = simtel_array_histograms_instance.trigger_rate_per_histogram(livetime=5)
    assert trigger_rate[0].unit == 1 / u.s
