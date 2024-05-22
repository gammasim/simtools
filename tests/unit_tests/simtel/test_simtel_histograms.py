#!/usr/bin/python3

import logging

import matplotlib.pyplot as plt
import pytest
from astropy.table import Table
from matplotlib.collections import QuadMesh

from simtools.io_operations.hdf5_handler import read_hdf5
from simtools.simtel.simtel_histogram import (
    InconsistentHistogramFormat,
    SimtelHistogram,
)
from simtools.simtel.simtel_histograms import SimtelHistograms

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def simtel_array_histograms_file(io_handler, corsika_output_file_name):
    return io_handler.get_input_data_file(
        file_name="run201_proton_za20deg_azm0deg_North_TestLayout_test-prod.simtel.zst",
        test=True,
    )


@pytest.fixture
def simtel_array_histograms_instance(simtel_array_histograms_file):
    instance = SimtelHistograms(
        histogram_files=[simtel_array_histograms_file, simtel_array_histograms_file], test=True
    )
    return instance


def test_file_does_not_exist(caplog):
    with caplog.at_level(logging.ERROR):
        with pytest.raises(FileNotFoundError):
            _ = SimtelHistogram(histogram_file="non_existent_file.simtel.zst")
    assert "does not exist." in caplog.text


def test_calculate_trigger_rates(simtel_array_histograms_instance, caplog):
    import astropy.units as u

    with caplog.at_level(logging.INFO):
        (
            sim_event_rates,
            triggered_event_rates,
            triggered_event_rate_uncertainties,
            trigger_rate_in_tables,
        ) = simtel_array_histograms_instance.calculate_trigger_rates(print_info=False)
        assert pytest.approx(sim_event_rates[0].value, 0.2) == 2e7
        assert sim_event_rates[0].unit == 1 / u.s
        assert pytest.approx(triggered_event_rate_uncertainties[0].value, 0.1) == 103
        assert trigger_rate_in_tables[0]["Energy (TeV)"][0] == 0.001 * u.TeV
    assert "Histogram" in caplog.text
    assert "Total number of simulated events" in caplog.text
    assert "Total number of triggered events" in caplog.text
    assert "Estimated equivalent observation time corresponding to the number of" in caplog.text
    assert "System trigger event rate" in caplog.text


def test_number_of_files(simtel_array_histograms_instance):
    assert simtel_array_histograms_instance.number_of_files == 2


def test_check_consistency(simtel_array_histograms_instance):
    first_hist_file = {"lower_x": 0, "upper_x": 10, "n_bins_x": 5, "title": "Histogram 1"}
    second_hist_file = {
        "lower_x": 0,
        "upper_x": 20,  # Different upper_x
        "n_bins_x": 5,
        "title": "Histogram 2",
    }

    # Test if the method raises InconsistentHistogramFormat exception
    with pytest.raises(InconsistentHistogramFormat):
        simtel_array_histograms_instance._check_consistency(first_hist_file, second_hist_file)


def test_meta_dict(simtel_array_histograms_instance):
    assert "simtools_version" in simtel_array_histograms_instance._meta_dict


def test_export_histograms(simtel_array_histograms_instance, io_handler):
    file_with_path = io_handler.get_output_directory(dir_type="test").joinpath(
        "test_hist_simtel.hdf5"
    )
    # Default values
    simtel_array_histograms_instance.export_histograms(file_with_path)

    assert file_with_path.exists()

    # Read simtel file
    list_of_tables = read_hdf5(file_with_path)
    assert len(list_of_tables) == 10
    for table in list_of_tables:
        assert isinstance(table, Table)


def test_list_of_histograms(simtel_array_histograms_instance):
    list_of_histograms = simtel_array_histograms_instance.list_of_histograms
    assert isinstance(list_of_histograms, list)
    assert len(list_of_histograms) == 2


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
