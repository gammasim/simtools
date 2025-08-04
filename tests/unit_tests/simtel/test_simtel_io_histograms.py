#!/usr/bin/python3

import copy
import logging
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.table import Table
from matplotlib.collections import QuadMesh

from simtools.io.hdf5_handler import read_hdf5
from simtools.simtel.simtel_io_histogram import (
    HistogramIdNotFoundError,
    InconsistentHistogramFormatError,
    SimtelIOHistogram,
)
from simtools.simtel.simtel_io_histograms import SimtelIOHistograms

logger = logging.getLogger()


@pytest.fixture
def sim_telarray_file_proton_list(io_handler, tmp_test_directory):
    output_file = tmp_test_directory / "simtel_output_files.txt"
    resource_dir = Path("./tests/resources")
    files = sorted(
        resource_dir.glob(
            "proton_run00020*_za20deg_azm000deg_North_alpha_6.0.0_test_file.simtel.zst"
        )
    )
    with output_file.open("w") as f:
        for file in files:
            f.write(f"{file}\n")
    return output_file


@pytest.fixture
def simtel_hists_hdata_io_instance(sim_telarray_hdata_file_gamma):
    return SimtelIOHistograms(
        histogram_files=sim_telarray_hdata_file_gamma, view_cone=[0, 10], energy_range=[0.001, 300]
    )


@pytest.fixture
def sim_telarray_histograms_instance(sim_telarray_file_proton):
    return SimtelIOHistograms(
        histogram_files=[sim_telarray_file_proton, sim_telarray_file_proton], test=True
    )


@pytest.fixture
def sim_telarray_histograms_instance_file_list(sim_telarray_file_proton_list):
    return SimtelIOHistograms(histogram_files=sim_telarray_file_proton_list, test=True)


def test_file_does_not_exist(caplog):
    with caplog.at_level(logging.ERROR):
        with pytest.raises(FileNotFoundError):
            _ = SimtelIOHistogram(histogram_file="non_existent_file.simtel.zst")
    assert "does not exist." in caplog.text


def test_calculate_trigger_rates(
    sim_telarray_histograms_instance,
    sim_telarray_histograms_instance_file_list,
    simtel_hists_hdata_io_instance,
    caplog,
):
    with caplog.at_level(logging.INFO):
        (
            sim_event_rates,
            triggered_event_rates,
            triggered_event_rate_uncertainties,
            trigger_rate_in_tables,
        ) = sim_telarray_histograms_instance.calculate_trigger_rates(print_info=False)
    assert pytest.approx(sim_event_rates[0].value, 0.2) == 2e7
    assert sim_event_rates[0].unit == 1 / u.s
    assert pytest.approx(triggered_event_rate_uncertainties[0].value, 0.1) == 11515
    assert trigger_rate_in_tables[0]["Energy (TeV)"][0] == 0.001 * u.TeV
    assert "Histogram" in caplog.text
    assert "Total number of simulated events" in caplog.text
    assert "Total number of triggered events" in caplog.text
    assert "Estimated equivalent observation time corresponding to the number of" in caplog.text
    assert "System trigger event rate" in caplog.text


def test_calculate_trigger_rates_file_list(sim_telarray_histograms_instance, caplog):
    with caplog.at_level(logging.INFO):
        (
            sim_event_rates,
            _,
            _,
            trigger_rate_in_tables,
        ) = sim_telarray_histograms_instance.calculate_trigger_rates(
            print_info=False, stack_files=True
        )
        second_sim_event_rates, _, _, _ = sim_telarray_histograms_instance._rates_for_each_file()
        assert second_sim_event_rates == [sim_event_rates, sim_event_rates]
    assert "System trigger event rate for stacked files" in caplog.text


def test_calculate_trigger_rates_hdata(simtel_hists_hdata_io_instance):
    # Test the trigger rate estimate for input hdata file
    (
        sim_event_rates,
        triggered_event_rates,
        triggered_event_rate_uncertainties,
        trigger_rate_in_tables,
    ) = simtel_hists_hdata_io_instance.calculate_trigger_rates(print_info=False)
    assert pytest.approx(sim_event_rates[0].value, 0.2) == 2.5e9
    assert sim_event_rates[0].unit == 1 / u.s
    assert pytest.approx(triggered_event_rate_uncertainties[0].value, 0.1) == 70000
    assert trigger_rate_in_tables[0]["Energy (TeV)"][0] == 0.001 * u.TeV


def test_rates_for_each_file(sim_telarray_histograms_instance):
    (
        sim_event_rate,
        triggered_event_rate,
        triggered_event_rate_uncertainty,
    ) = sim_telarray_histograms_instance._rates_for_stacked_files()
    assert isinstance(sim_event_rate, list)
    assert pytest.approx(triggered_event_rate_uncertainty[0].value, 0.1) == 8142  # uncertainty
    # decreased by stacking files


def test_fill_stacked_events(sim_telarray_histograms_instance, caplog):
    new_instance = copy.copy(sim_telarray_histograms_instance)
    new_instance.combined_hists
    # Test defect histograms
    for histogram_index, hist in enumerate(
        new_instance.combined_hists
    ):  # altering intentionally the ids
        if hist["id"] == 1:
            new_instance.combined_hists[histogram_index]["id"] = 99
        if hist["id"] == 2:
            new_instance.combined_hists[histogram_index]["id"] = 99
    with caplog.at_level(logging.ERROR):
        with pytest.raises(HistogramIdNotFoundError):
            new_instance._fill_stacked_events()


def test_get_stacked_num_events(sim_telarray_histograms_instance):
    ref_n_sim, ref_n_tri = sim_telarray_histograms_instance.get_stacked_num_events()
    n_sim, n_tri = 0, 0
    for _, hist in enumerate(sim_telarray_histograms_instance.combined_hists):
        if hist["id"] == 1:
            n_sim += np.sum(hist["data"])
        if hist["id"] == 2:
            n_tri += np.sum(hist["data"])
    assert ref_n_sim == n_sim
    assert ref_n_tri == n_tri


def test_number_of_files(sim_telarray_histograms_instance):
    assert sim_telarray_histograms_instance.number_of_files == 2


def test_check_consistency(sim_telarray_histograms_instance):
    first_hist_file = {"lower_x": 0, "upper_x": 10, "n_bins_x": 5, "title": "Histogram 1"}
    second_hist_file = {
        "lower_x": 0,
        "upper_x": 20,  # Different upper_x
        "n_bins_x": 5,
        "title": "Histogram 2",
    }

    # Test if the method raises InconsistentHistogramFormatError exception
    with pytest.raises(InconsistentHistogramFormatError):
        sim_telarray_histograms_instance._check_consistency(first_hist_file, second_hist_file)


def test_meta_dict(sim_telarray_histograms_instance):
    assert "simtools_version" in sim_telarray_histograms_instance._meta_dict


def test_export_histograms(sim_telarray_histograms_instance, io_handler):
    file_with_path = io_handler.get_output_directory().joinpath("test_hist_simtel.hdf5")
    # Default values
    sim_telarray_histograms_instance.export_histograms(file_with_path)

    assert file_with_path.exists()

    # Read sim_telarray file
    list_of_tables = read_hdf5(file_with_path)
    assert len(list_of_tables) == 13
    for table in list_of_tables:
        assert isinstance(table, Table)


def test_list_of_histograms(sim_telarray_histograms_instance):
    list_of_histograms = sim_telarray_histograms_instance.list_of_histograms
    assert isinstance(list_of_histograms, list)
    assert len(list_of_histograms) == 2


def test_combine_histogram_files(sim_telarray_file_proton, caplog):
    # Reading one histogram file
    instance_alone = SimtelIOHistograms(histogram_files=sim_telarray_file_proton, test=True)

    # Passing the same file twice
    instance_all = SimtelIOHistograms(
        histogram_files=[sim_telarray_file_proton, sim_telarray_file_proton], test=True
    )
    assert (
        2 * instance_alone.combined_hists[0]["data"] == instance_all.combined_hists[0]["data"]
    ).all()

    # Test inconsistency
    instance_all.combined_hists[0]["lower_x"] = instance_all.combined_hists[0]["lower_x"] + 1
    instance_all._combined_hists = None
    assert instance_all._combined_hists is None
    with caplog.at_level("ERROR"):
        with pytest.raises(InconsistentHistogramFormatError):
            _ = instance_all.combined_hists
    assert "Trying to add histograms with inconsistent dimensions" in caplog.text


def test_plot_one_histogram(sim_telarray_histograms_instance):
    fig, ax = plt.subplots()
    sim_telarray_histograms_instance.plot_one_histogram(0, ax)
    quadmesh = ax.collections[0]
    assert isinstance(quadmesh, QuadMesh)
