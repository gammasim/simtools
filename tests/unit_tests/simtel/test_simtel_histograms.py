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
    SimtelHistogram,
    SimtelHistograms,
)

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


@pytest.fixture
def simtel_array_histogram_instance(simtel_array_histograms_file):
    instance = SimtelHistogram(histogram_file=simtel_array_histograms_file)
    return instance


def test_number_of_histogram_types(simtel_array_histogram_instance):
    assert simtel_array_histogram_instance.number_of_histogram_types == 10


def test_get_histogram_type_title(simtel_array_histogram_instance):
    title = simtel_array_histogram_instance.get_histogram_type_title(i_hist=0)
    assert title == "Events, without weights (Ra, log10(E))"


def test_config(simtel_array_histogram_instance):
    config = simtel_array_histogram_instance.config
    assert config is not None
    assert "B_declination" in config


def test_total_num_simulated_events(simtel_array_histogram_instance):
    total_num_simulated_events = simtel_array_histogram_instance.total_num_simulated_events
    assert total_num_simulated_events == 2000


def test_total_num_triggered_events(simtel_array_histogram_instance):
    total_num_triggered_events = simtel_array_histogram_instance.total_num_triggered_events
    assert total_num_triggered_events == 1.0


def test_fill_event_histogram_dicts(simtel_array_histogram_instance):
    (
        events_histogram,
        triggered_events_histogram,
    ) = simtel_array_histogram_instance.fill_event_histogram_dicts()
    assert events_histogram is not None
    assert events_histogram["content_inside"] == 2000.0
    assert triggered_events_histogram is not None
    assert triggered_events_histogram["content_inside"] == 1.0


def test_view_cone(simtel_array_histogram_instance):
    view_cone = simtel_array_histogram_instance.view_cone
    assert (view_cone == [0, 10] * u.deg).all()


def test_total_area(simtel_array_histogram_instance):
    total_area = simtel_array_histogram_instance.total_area
    assert total_area.unit == u.cm**2
    assert pytest.approx(total_area.value, 2) == 1.25e11


def test_energy_range(simtel_array_histogram_instance):
    energy_range = simtel_array_histogram_instance.energy_range
    assert energy_range[0].unit == u.TeV
    assert energy_range[1].value > energy_range[0].value


def test_get_correction_factor(simtel_array_histogram_instance):
    correction_factor = simtel_array_histogram_instance.get_correction_factor()
    assert pytest.approx(correction_factor.value, 0.1) == 10635


def test_get_simulation_spectral_distribution(simtel_array_histogram_instance):
    simulation_spectral_distribution = (
        simtel_array_histogram_instance.get_simulation_spectral_distribution()
    )
    assert simulation_spectral_distribution is not None


def test_estimate_observation_time(simtel_array_histogram_instance):
    observation_time = simtel_array_histogram_instance.estimate_observation_time()
    assert observation_time.unit == u.s
    assert pytest.approx(observation_time.value, 0.1) == 9.4e-5


def test_estimate_trigger_rate_uncertainty(simtel_array_histogram_instance):
    trigger_rate_uncertainty = simtel_array_histogram_instance._estimate_trigger_rate_uncertainty()
    assert trigger_rate_uncertainty.unit == 1 / u.s
    assert pytest.approx(trigger_rate_uncertainty.value, 0.1) == 10635


def test_trigger_rate_per_histogram(simtel_array_histogram_instance):
    trigger_rate = simtel_array_histogram_instance.trigger_rate_per_histogram(re_weight=True)
    assert pytest.approx(trigger_rate[0].value, 0.1) == 3.45
    assert trigger_rate[0].unit == 1 / u.s


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
