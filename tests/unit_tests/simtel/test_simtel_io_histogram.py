#!/usr/bin/python3

import copy
import logging

import numpy as np
import pytest
from astropy import units as u
from ctao_cr_spectra.definitions import IRFDOC_PROTON_SPECTRUM
from ctao_cr_spectra.spectral import PowerLaw

from simtools.simtel.simtel_io_histogram import HistogramIdNotFoundError, SimtelIOHistogram

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture()
def simtel_array_histograms_file(io_handler, corsika_output_file_name):
    return io_handler.get_input_data_file(
        file_name="run201_proton_za20deg_azm0deg_North_test_layout_test-prod.simtel.zst",
        test=True,
    )


@pytest.fixture()
def simtel_array_histogram_instance(simtel_array_histograms_file):
    instance = SimtelIOHistogram(histogram_file=simtel_array_histograms_file)
    return instance


def test_file_does_not_exist(caplog):
    with caplog.at_level(logging.ERROR):
        with pytest.raises(FileNotFoundError):
            _ = SimtelIOHistogram(histogram_file="non_existent_file.simtel.zst")
    assert "does not exist." in caplog.text


def test_number_of_histogram_types(simtel_array_histogram_instance):
    assert simtel_array_histogram_instance.number_of_histogram_types == 10


def test_get_histogram_type_title(simtel_array_histogram_instance):
    title = simtel_array_histogram_instance.get_histogram_type_title(histogram_index=0)
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


def test_fill_event_histogram_dicts(simtel_array_histogram_instance, caplog):
    (
        events_histogram,
        triggered_events_histogram,
    ) = simtel_array_histogram_instance.fill_event_histogram_dicts()
    assert events_histogram is not None
    assert events_histogram["content_inside"] == 2000.0
    assert triggered_events_histogram is not None
    assert triggered_events_histogram["content_inside"] == 1.0

    # Test defect histograms
    new_instance = copy.copy(simtel_array_histogram_instance)
    for histogram_index, hist in enumerate(
        new_instance.histogram
    ):  # altering intentionally the ids
        if hist["id"] == 1:
            new_instance.histogram[histogram_index]["id"] = 99
        if hist["id"] == 2:
            new_instance.histogram[histogram_index]["id"] = 99
    with caplog.at_level(logging.ERROR):
        with pytest.raises(HistogramIdNotFoundError):
            new_instance.fill_event_histogram_dicts()
    assert "Histograms ids not found. Please check your files." in caplog.text


def test_produce_triggered_to_sim_fraction_hist(simtel_array_histogram_instance):
    events_histogram = {"data": np.array([[10, 20, 30], [40, 50, 60]])}
    triggered_events_histogram = {"data": np.array([[5, 10, 15], [20, 25, 30]])}

    result = simtel_array_histogram_instance._produce_triggered_to_sim_fraction_hist(
        events_histogram, triggered_events_histogram
    )
    expected_result = np.array([0.5, 0.5])

    assert result == pytest.approx(expected_result, 0.01)

    # Additional assertions
    assert np.all(result >= 0)  # Check if all values are non-negative
    assert np.all(result <= 1)  # Check if all values are less than or equal to 1


def test_initialize_histogram_axes(simtel_array_histogram_instance):
    events_histogram = {
        "lower_x": 0,
        "upper_x": 10,
        "n_bins_x": 5,
        "lower_y": 1,
        "upper_y": 3,
        "n_bins_y": 4,
    }
    simtel_array_histogram_instance._initialize_histogram_axes(events_histogram)
    expected_radius_axis = np.linspace(0, 10, 6)
    expected_energy_axis = np.logspace(1, 3, 5)
    assert np.array_equal(simtel_array_histogram_instance.radius_axis, expected_radius_axis)
    assert np.array_equal(simtel_array_histogram_instance.energy_axis, expected_energy_axis)


def test_get_particle_distribution_function(simtel_array_histogram_instance):
    # Test reference distribution function
    reference_function = simtel_array_histogram_instance.get_particle_distribution_function(
        label="reference"
    )
    assert isinstance(
        reference_function, PowerLaw
    )  # Assuming PowerLaw is the expected type for the reference function

    # Test simulation distribution function
    simulation_function = simtel_array_histogram_instance.get_particle_distribution_function(
        label="simulation"
    )
    assert simulation_function.index == -2.0

    # Test invalid label
    with pytest.raises(ValueError):
        simtel_array_histogram_instance.get_particle_distribution_function(label="invalid_label")


def test_integrate_in_energy_bin(simtel_array_histogram_instance):
    result = simtel_array_histogram_instance._integrate_in_energy_bin(
        IRFDOC_PROTON_SPECTRUM, np.array([1, 10])
    )
    assert pytest.approx(result.value, 0.1) == 5.9e-06


def test_view_cone(simtel_array_histogram_instance):
    view_cone = simtel_array_histogram_instance.view_cone
    assert (view_cone == [0, 10] * u.deg).all()


def test_compute_system_trigger_rate_and_table(simtel_array_histogram_instance):

    assert simtel_array_histogram_instance.trigger_rate is None
    assert simtel_array_histogram_instance.trigger_rate_uncertainty is None
    assert simtel_array_histogram_instance.trigger_rate_per_energy_bin is None
    simtel_array_histogram_instance.compute_system_trigger_rate()
    assert pytest.approx(simtel_array_histogram_instance.trigger_rate.value, 0.1) == 9006
    assert simtel_array_histogram_instance.trigger_rate.unit == 1 / u.s
    assert (
        pytest.approx(simtel_array_histogram_instance.trigger_rate_uncertainty.value, 0.1) == 9008
    )
    assert simtel_array_histogram_instance.trigger_rate_uncertainty.unit == 1 / u.s
    assert len(simtel_array_histogram_instance.trigger_rate_per_energy_bin.value) == 120

    events_histogram, triggered_events_histogram = (
        simtel_array_histogram_instance.fill_event_histogram_dicts()
    )
    simtel_array_histogram_instance._initialize_histogram_axes(events_histogram)
    table = simtel_array_histogram_instance.trigger_info_in_table()
    assert table["Energy (TeV)"].value[0] == 1.00000000e-03


def test_compute_system_trigger_rate_with_input(simtel_array_histogram_instance):

    new_instance = copy.copy(simtel_array_histogram_instance)
    events_histogram, triggered_events_histogram = new_instance.fill_event_histogram_dicts()
    new_events_histogram = copy.copy(events_histogram)
    new_events_histogram["data"] = 1e-9 * new_events_histogram["data"]
    new_triggered_events_histogram = copy.copy(triggered_events_histogram)
    new_triggered_events_histogram["data"] = 1e-8 * new_triggered_events_histogram["data"]
    new_instance.compute_system_trigger_rate(new_events_histogram, new_triggered_events_histogram)
    assert pytest.approx(new_instance.trigger_rate.value, 0.1) == 9006 * 10
    assert new_instance.trigger_rate.unit == 1 / u.s
    assert pytest.approx(new_instance.trigger_rate_uncertainty.value, 0.1) == 902892137
    assert new_instance.trigger_rate_uncertainty.unit == 1 / u.s


def test_produce_trigger_meta_data(simtel_array_histogram_instance, simtel_array_histograms_file):

    trigger_rate = 1000  # Hz

    simtel_array_histogram_instance.histogram_file = simtel_array_histograms_file
    simtel_array_histogram_instance.trigger_rate = (
        trigger_rate * u.Hz
    )  # Convert to astropy Quantity

    result = simtel_array_histogram_instance.produce_trigger_meta_data()

    expected_result = {
        "simtel_array_file": simtel_array_histograms_file,
        "simulation_input": simtel_array_histogram_instance.print_info(mode="silent"),
        "system_trigger_rate (Hz)": trigger_rate,
    }
    assert result == expected_result


def test_print_info(simtel_array_histogram_instance):
    info_dict = simtel_array_histogram_instance.print_info(mode=None)
    assert "view_cone" in info_dict
    assert "energy_range" in info_dict


def test_total_area(simtel_array_histogram_instance, simtel_array_histograms_file):
    total_area = simtel_array_histogram_instance.total_area
    assert total_area.unit == u.cm**2
    assert pytest.approx(total_area.value, 0.05) == 1.25e11
    new_instance = SimtelIOHistogram(
        histogram_file=simtel_array_histograms_file, area_from_distribution=True
    )
    assert pytest.approx(new_instance.total_area.value, 0.05) == 1.3e11


def test_energy_range(simtel_array_histogram_instance):
    energy_range = simtel_array_histogram_instance.energy_range
    assert energy_range[0].unit == u.TeV
    assert energy_range[1].value > energy_range[0].value


def test_solid_angle(simtel_array_histogram_instance):
    solid_angle = simtel_array_histogram_instance.solid_angle
    assert pytest.approx(solid_angle.value, 0.1) == 0.095


def test_estimate_observation_time(simtel_array_histogram_instance):
    observation_time = simtel_array_histogram_instance.estimate_observation_time()
    assert observation_time.unit == u.s
    assert pytest.approx(observation_time.value, 0.1) == 9.4e-5
    observation_time = simtel_array_histogram_instance.estimate_observation_time(
        stacked_num_simulated_events=100
    )
    assert observation_time.unit == u.s
    assert pytest.approx(observation_time.value, 0.1) == 4.7e-6


def test_estimate_trigger_rate_uncertainty(simtel_array_histogram_instance):

    simtel_array_histogram_instance.compute_system_trigger_rate()
    trigger_rate_uncertainty = simtel_array_histogram_instance.estimate_trigger_rate_uncertainty(
        simtel_array_histogram_instance.trigger_rate,
        simtel_array_histogram_instance.total_num_simulated_events,
        simtel_array_histogram_instance.total_num_triggered_events,
    )
    assert trigger_rate_uncertainty.unit == 1 / u.s
    assert pytest.approx(trigger_rate_uncertainty.value, 0.1) == 9008
    trigger_rate_uncertainty = simtel_array_histogram_instance.estimate_trigger_rate_uncertainty(
        1e4 / u.s, 100, 10
    )
    assert trigger_rate_uncertainty.unit == 1 / u.s
    assert pytest.approx(trigger_rate_uncertainty.value, 0.1) == 3316
