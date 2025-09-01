#!/usr/bin/python3


import copy
import logging
from pathlib import Path

import boost_histogram as bh
import numpy as np
import pytest
from astropy import units as u
from astropy.table import Table

from simtools import version
from simtools.corsika.corsika_histograms import (
    CorsikaHistograms,
    HistogramNotCreatedError,
)
from simtools.io.hdf5_handler import read_hdf5

x_axis_string = "x axis"
y_axis_string = "y axis"
z_axis_string = "z axis"


def test_init(corsika_histograms_instance, corsika_output_file_name):
    assert corsika_histograms_instance.input_file.name == Path(corsika_output_file_name).name
    with pytest.raises(FileNotFoundError):
        CorsikaHistograms("wrong_file_name")
    assert len(corsika_histograms_instance.event_information) > 15
    assert "zenith" in corsika_histograms_instance.event_information


def test_version(corsika_histograms_instance):
    assert corsika_histograms_instance.corsika_version == pytest.approx(7.741)


def test_initialize_header(corsika_histograms_instance):
    corsika_histograms_instance._initialize_header()
    # Check the some elements of the header
    manual_header = {
        "run_number": 1 * u.dimensionless_unscaled,
        "date": 230208 * u.dimensionless_unscaled,
        "version": 7.741 * u.dimensionless_unscaled,
        "n_observation_levels": 1 * u.dimensionless_unscaled,
        "observation_height": [214700.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] * u.cm,
        "energy_min": 10 * u.GeV,
    }
    assert len(corsika_histograms_instance.header) > 10
    for key in manual_header:
        assert (
            pytest.approx(corsika_histograms_instance.header[key].value) == manual_header[key].value
        )


def test_telescope_indices(corsika_histograms_instance):
    corsika_histograms_instance.telescope_indices = [0, 1, 2]
    assert (corsika_histograms_instance.telescope_indices == [0, 1, 2]).all()
    # Test int as input
    corsika_histograms_instance.telescope_indices = 1
    assert corsika_histograms_instance.telescope_indices == [1]
    # Test non-integer indices
    with pytest.raises(TypeError):
        corsika_histograms_instance.telescope_indices = [1.5, 2, 2.5]


def test_read_event_information(corsika_histograms_instance):
    # Check some of the event info
    manual_event_info = {
        "event_number": [1, 2] * u.dimensionless_unscaled,
        "particle_id": [1, 1] * u.dimensionless_unscaled,
        "total_energy": [10, 10] * u.GeV,
        "starting_altitude": [0, 0] * (u.g / (u.cm**2)),
    }
    for key in manual_event_info:
        assert (corsika_histograms_instance.event_information[key] == manual_event_info[key]).all()


def test_get_header_astropy_units(corsika_histograms_instance):
    parameters = ["momentum_x", "event_number", "starting_altitude", "total_energy"]
    non_astropy_units = ["GeV/c", "", "g/cm2", "GeV"]
    all_event_astropy_units = corsika_histograms_instance._get_header_astropy_units(
        parameters, non_astropy_units
    )
    for astropy_unit in all_event_astropy_units.values():
        assert isinstance(astropy_unit, u.core.CompositeUnit)


def test_hist_config_default_config(corsika_histograms_instance, caplog):
    with caplog.at_level("WARNING"):
        hist_config = corsika_histograms_instance.hist_config
    assert "No histogram configuration was defined before." in caplog.text
    assert isinstance(hist_config, dict)
    assert hist_config == corsika_histograms_instance._create_histogram_default_config()


def test_hist_config_custom_config(corsika_histograms_instance):
    custom_config = {
        "hist_position": {
            x_axis_string: {"bins": 100, "start": -1000, "stop": 1000, "scale": "linear"},
            y_axis_string: {"bins": 100, "start": -1000, "stop": 1000, "scale": "linear"},
            z_axis_string: {"bins": 80, "start": 200, "stop": 1000, "scale": "linear"},
        },
        "hist_direction": {
            "azimuth": {"bins": 36, "start": 0, "stop": 360, "scale": "linear"},
            "zenith": {"bins": 18, "start": 0, "stop": 90, "scale": "linear"},
        },
    }
    corsika_histograms_instance._hist_config = custom_config
    hist_config = corsika_histograms_instance.hist_config
    assert hist_config == custom_config


def test_hist_config_save_and_read_yml(corsika_histograms_instance, io_handler):
    # Test producing the yaml file
    temp_hist_config = corsika_histograms_instance.hist_config
    corsika_histograms_instance.hist_config_to_yaml()
    output_file = corsika_histograms_instance.output_path.joinpath("hist_config.yml")
    corsika_histograms_instance.hist_config = output_file
    assert corsika_histograms_instance.hist_config == temp_hist_config


def test_create_regular_axes_valid_label(corsika_histograms_instance):
    hists = ["hist_position", "hist_direction", "hist_time_altitude"]
    num_of_axes = [3, 2, 2]
    for i_hist, _ in enumerate(hists):
        axes = corsika_histograms_instance._create_regular_axes(hists[i_hist])
        assert len(axes) == num_of_axes[i_hist]
        for i_axis in range(num_of_axes[i_hist]):
            assert isinstance(axes[i_axis], bh.axis.Regular)


def test_create_regular_axes_invalid_label(corsika_histograms_instance):
    label = "invalid_label"
    with pytest.raises(ValueError, match=r"allowed labels must be one of the following"):
        corsika_histograms_instance._create_regular_axes(label)


def test_create_histograms(corsika_histograms_instance):
    # Test once for individual_telescopes True and once false
    individual_telescopes = [True, False]
    corsika_histograms_instance.telescope_indices = [0, 1, 2, 3]
    num_of_expected_hists = [4, 1]
    for i_test in range(2):
        corsika_histograms_instance._create_histograms(individual_telescopes[i_test])
        assert corsika_histograms_instance.num_of_hist == num_of_expected_hists[i_test]
        assert len(corsika_histograms_instance.hist_position) == num_of_expected_hists[i_test]
        assert len(corsika_histograms_instance.hist_direction) == num_of_expected_hists[i_test]
        assert len(corsika_histograms_instance.hist_time_altitude) == num_of_expected_hists[i_test]
        assert isinstance(corsika_histograms_instance.hist_position[0], bh.Histogram)
        assert isinstance(corsika_histograms_instance.hist_direction[0], bh.Histogram)
        assert isinstance(corsika_histograms_instance.hist_time_altitude[0], bh.Histogram)


def test_fill_histograms_no_rotation(corsika_output_file_name, io_handler):
    # Sample test of photons: 1 telescope, 2 photons
    photons = [
        {
            "x": 722.6629,
            "y": 972.66925,
            "cx": 0.34438822,
            "cy": -0.01040812,
            "time": -115.58354,
            "zem": 1012681.8,
            "photons": 2.2303813,
            "wavelength": -428.27454,
        },
        {
            "x": 983.2037,
            "y": 809.2618,
            "cosx": 0.34259367,
            "cosy": -0.00547006,
            "time": -113.36441,
            "zem": 1193760.8,
            "photons": 2.4614816,
            "wavelength": -522.7789,
        },
    ]

    corsika_histograms_instance_fill = CorsikaHistograms(corsika_output_file_name)
    corsika_histograms_instance_fill.individual_telescopes = False
    corsika_histograms_instance_fill.telescope_indices = [0]

    corsika_histograms_instance_fill._create_histograms(individual_telescopes=False)

    # No count in the histogram before filling it
    assert np.count_nonzero(corsika_histograms_instance_fill.hist_direction[0].values()) == 0
    corsika_histograms_instance_fill._fill_histograms(
        photons, rotation_around_z_axis=None, rotation_around_y_axis=None
    )
    # At least one count in the histogram after filling it
    assert np.count_nonzero(corsika_histograms_instance_fill.hist_direction[0].values()) > 0


def test_get_hist_1d_projection(corsika_histograms_instance_set_histograms, caplog):
    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError, match="label_not_valid is not valid."):
            corsika_histograms_instance_set_histograms._get_hist_1d_projection("label_not_valid")
    assert "label_not_valid is not valid." in caplog.text

    labels = ["wavelength", "time", "altitude"]
    expected_shape_of_bin_edges = [(1, 81), (1, 101), (1, 101)]
    expected_shape_of_values = [(1, 80), (1, 100), (1, 100)]
    expected_mean = [125.4, 116.3, 116.3]
    expected_std = [153.4, 378.2, 483.8]
    for i_hist, hist_label in enumerate(labels):
        (
            hist_1d_list,
            x_bin_edges_list,
        ) = corsika_histograms_instance_set_histograms._get_hist_1d_projection(hist_label)
        assert np.shape(x_bin_edges_list) == expected_shape_of_bin_edges[i_hist]
        assert np.shape(hist_1d_list) == expected_shape_of_values[i_hist]
    assert np.mean(hist_1d_list) == pytest.approx(expected_mean[i_hist], abs=1e-1)
    assert np.std(hist_1d_list) == pytest.approx(expected_std[i_hist], abs=1e-1)


def test_set_histograms_all_telescopes_1_histogram(corsika_histograms_instance):
    # all telescopes, but 1 histogram
    corsika_histograms_instance.set_histograms(telescope_indices=None, individual_telescopes=False)
    assert np.shape(corsika_histograms_instance.hist_position[0].values()) == (100, 100, 80)
    # assert that the histograms are filled
    assert np.count_nonzero(corsika_histograms_instance.hist_position[0][:, :, sum].view().T) == 159
    # and the sum is what we expect
    assert np.sum(corsika_histograms_instance.hist_position[0].view()) == pytest.approx(10031.0)


def test_set_histograms_3_telescopes_1_histogram(corsika_histograms_instance):
    # 3 telescopes, but 1 histogram
    corsika_histograms_instance.set_histograms(
        telescope_indices=[0, 1, 2], individual_telescopes=False
    )
    assert np.shape(corsika_histograms_instance.hist_position[0].values()) == (100, 100, 80)
    # assert that the histograms are filled
    assert np.count_nonzero(corsika_histograms_instance.hist_position[0][:, :, sum].view().T) == 12
    # and the sum is what we expect
    assert np.sum(corsika_histograms_instance.hist_position[0].view()) == pytest.approx(3177.0)


def test_set_histograms_3_telescopes_3_histograms(corsika_histograms_instance):
    # 3 telescopes and 3 histograms
    corsika_histograms_instance.set_histograms(
        telescope_indices=[0, 1, 2], individual_telescopes=True
    )

    hist_non_zero_bins = [827, 911, 966]
    hist_sum = [959.0, 1062.0, 1156.0]
    for i_hist in range(3):
        assert np.shape(corsika_histograms_instance.hist_position[i_hist].values()) == (64, 64, 80)
        # assert that the histograms are filled
        assert (
            np.count_nonzero(corsika_histograms_instance.hist_position[i_hist][:, :, sum].view().T)
            == hist_non_zero_bins[i_hist]
        )
        # and the sum is what we expect
        assert np.sum(corsika_histograms_instance.hist_position[i_hist].view()) == pytest.approx(
            hist_sum[i_hist]
        )


def test_set_histograms_passing_config(corsika_histograms_instance):
    new_hist_config = copy.copy(corsika_histograms_instance.hist_config)
    xy_maximum = 500 * u.m
    xy_bin = 100
    new_hist_config["hist_position"] = {
        x_axis_string: {
            "bins": xy_bin,
            "start": -xy_maximum,
            "stop": xy_maximum,
            "scale": "linear",
        },
        y_axis_string: {
            "bins": xy_bin,
            "start": -xy_maximum,
            "stop": xy_maximum,
            "scale": "linear",
        },
        z_axis_string: {
            "bins": 80,
            "start": 200 * u.nm,
            "stop": 1000 * u.nm,
            "scale": "linear",
        },
    }
    corsika_histograms_instance.set_histograms(
        individual_telescopes=False, hist_config=new_hist_config
    )
    assert corsika_histograms_instance.hist_position[0][:, :, sum].shape == (100, 100)
    assert corsika_histograms_instance.hist_position[0][:, :, sum].axes[0].edges[0] == -500
    assert corsika_histograms_instance.hist_position[0][:, :, sum].axes[0].edges[-1] == 500


def test_raise_if_no_histogram(corsika_output_file_name, caplog, io_handler):
    corsika_histograms_instance_not_hist = CorsikaHistograms(corsika_output_file_name)
    with pytest.raises(HistogramNotCreatedError):
        corsika_histograms_instance_not_hist._raise_if_no_histogram()
    assert "The histograms were not created." in caplog.text


def test_get_hist_2d_projection(corsika_histograms_instance, caplog):
    corsika_histograms_instance.set_histograms()

    label = "hist_non_existent"
    with pytest.raises(ValueError, match=r"^label is not valid. Valid entries are"):
        corsika_histograms_instance._get_hist_2d_projection(label)
    assert "label is not valid." in caplog.text

    labels = ["counts", "density", "direction", "time_altitude"]
    hist_sums = [11633, 29.1, 11634, 11634]  # sum of photons are approximately the same
    # (except for the density hist, which is divided by the area)
    for i_label, label in enumerate(labels):
        hist_values, x_bin_edges, y_bin_edges = corsika_histograms_instance._get_hist_2d_projection(
            label
        )
        assert np.shape(x_bin_edges) == (1, 101)
        assert np.shape(y_bin_edges) == (1, 101)
        assert np.shape(hist_values) == (1, 100, 100)
    assert np.sum(hist_values) == pytest.approx(hist_sums[i_label], abs=1e-2)

    # Repeat the test for fewer telescopes and see that less photons are counted in
    corsika_histograms_instance.set_histograms(telescope_indices=[0, 1, 2])
    hist_sums = [3677, 9.2, 3677, 3677]
    for i_label, label in enumerate(labels):
        hist_values, x_bin_edges, y_bin_edges = corsika_histograms_instance._get_hist_2d_projection(
            label
        )
    assert np.sum(hist_values) == pytest.approx(hist_sums[i_label], abs=1e-2)


def test_get_2d_photon_position_distr(corsika_histograms_instance_set_histograms):
    density = corsika_histograms_instance_set_histograms.get_2d_photon_density_distr()

    # Test the values of the histogram
    assert np.sum(density[0]) == pytest.approx(29, abs=1e-1)
    counts = corsika_histograms_instance_set_histograms.get_2d_photon_position_distr()
    assert np.sum(counts[0]) == pytest.approx(11633, abs=1e-1)

    # The bin edges should be the same
    assert (counts[1] == density[1]).all()
    assert (counts[2] == density[2]).all()


def test_get_2d_photon_direction_distr(corsika_histograms_instance_set_histograms):
    for returned_variable in range(3):
        assert (
            corsika_histograms_instance_set_histograms.get_2d_photon_direction_distr()[
                returned_variable
            ]
            == corsika_histograms_instance_set_histograms._get_hist_2d_projection("direction")[
                returned_variable
            ]
        ).all()


def test_get_2d_photon_time_altitude_distr(corsika_histograms_instance_set_histograms):
    for returned_variable in range(3):
        assert (
            corsika_histograms_instance_set_histograms.get_2d_photon_time_altitude_distr()[
                returned_variable
            ]
            == corsika_histograms_instance_set_histograms._get_hist_2d_projection("time_altitude")[
                returned_variable
            ]
        ).all()


def test_get_2d_num_photons_distr(corsika_histograms_instance_set_histograms):
    corsika_histograms_instance_set_histograms.set_histograms(telescope_indices=[0, 4, 10])
    (
        num_photons_per_event_per_telescope,
        num_events_array,
        telescope_indices_array,
    ) = corsika_histograms_instance_set_histograms.get_2d_num_photons_distr()
    assert np.shape(num_events_array) == (1, 3)  # number of events in this output file + 1
    # (bin edges of hist)
    assert (telescope_indices_array == [0, 1, 2, 3]).all()
    assert num_photons_per_event_per_telescope[0][0, 0] == pytest.approx(2543.3, abs=5e-1)
    # 1st tel, 1st event
    assert num_photons_per_event_per_telescope[0][0, 1] == pytest.approx(290.4, abs=5e-1)
    # 1st tel, 2nd event
    assert num_photons_per_event_per_telescope[0][1, 0] == pytest.approx(1741, abs=5e-1)
    # 2nd tel, 1st event
    assert num_photons_per_event_per_telescope[0][1, 1] == pytest.approx(85.9, abs=5e-1)
    # 2nd tel, 2nd event


def test_get_photon_altitude_distr(corsika_histograms_instance_set_histograms):
    for returned_variable in range(2):
        assert (
            corsika_histograms_instance_set_histograms._get_hist_1d_projection("altitude")[
                returned_variable
            ]
            == corsika_histograms_instance_set_histograms.get_photon_altitude_distr()[
                returned_variable
            ]
        ).all()


def test_get_photon_time_of_emission_distr(corsika_histograms_instance_set_histograms):
    for returned_variable in range(2):
        assert (
            corsika_histograms_instance_set_histograms._get_hist_1d_projection("time")[
                returned_variable
            ]
            == corsika_histograms_instance_set_histograms.get_photon_time_of_emission_distr()[
                returned_variable
            ]
        ).all()


def test_get_photon_wavelength_distr(corsika_histograms_instance_set_histograms):
    for returned_variable in range(2):
        assert (
            corsika_histograms_instance_set_histograms._get_hist_1d_projection("wavelength")[
                returned_variable
            ]
            == corsika_histograms_instance_set_histograms.get_photon_wavelength_distr()[
                returned_variable
            ]
        ).all()


def test_get_photon_radial_distr_individual_telescopes(corsika_histograms_instance_set_histograms):
    # Individual telescopes
    corsika_histograms_instance_set_histograms.set_histograms(
        telescope_indices=[0, 1, 2], individual_telescopes=True, hist_config=None
    )
    _, x_bin_edges_list = corsika_histograms_instance_set_histograms.get_photon_radial_distr()
    for i_hist, _ in enumerate(corsika_histograms_instance_set_histograms.telescope_indices):
        assert np.amax(x_bin_edges_list[i_hist]) == 16
        assert np.size(x_bin_edges_list[i_hist]) == 33


def test_get_photon_radial_distr_some_telescopes(corsika_histograms_instance_set_histograms):
    # All given telescopes together
    corsika_histograms_instance_set_histograms.set_histograms(
        telescope_indices=[0, 1, 2, 3, 4, 5], individual_telescopes=False, hist_config=None
    )
    _, x_bin_edges_list = corsika_histograms_instance_set_histograms.get_photon_radial_distr()
    assert np.amax(x_bin_edges_list) == 1000
    assert np.size(x_bin_edges_list) == 51


def test_get_photon_radial_distr_input_some_tel_and_density(
    corsika_histograms_instance_set_histograms,
):
    # Retrieve input values
    corsika_histograms_instance_set_histograms.set_histograms(
        telescope_indices=None, individual_telescopes=False, hist_config=None
    )

    (
        hist_1d_list,
        x_bin_edges_list,
    ) = corsika_histograms_instance_set_histograms.get_photon_radial_distr(bins=100, max_dist=1200)
    assert np.amax(x_bin_edges_list) == 1200
    assert np.size(x_bin_edges_list) == 101

    # Test if the keyword density changes the output histogram but not the bin_edges
    (
        hist_1d_list_dens,
        x_bin_edges_list_dens,
    ) = corsika_histograms_instance_set_histograms.get_photon_density_distr(
        bins=100,
        max_dist=1200,
    )
    assert (x_bin_edges_list_dens == x_bin_edges_list).all()

    assert np.sum(hist_1d_list_dens) == pytest.approx(1.86, abs=1e-1)
    # density smaller because it divides
    # by the area (not counts per bin)
    assert np.sum(hist_1d_list) == pytest.approx(744.17, abs=3e-0)


def test_get_photon_radial_distr_input_all_tel(corsika_histograms_instance):
    corsika_histograms_instance.set_histograms(
        telescope_indices=[0, 1, 2, 3, 4, 5], individual_telescopes=True
    )

    # Default input values
    _, x_bin_edges_list = corsika_histograms_instance.get_photon_radial_distr()
    for i_tel, _ in enumerate(corsika_histograms_instance.telescope_indices):
        assert np.amax(x_bin_edges_list[i_tel]) == 16
        assert np.size(x_bin_edges_list[i_tel]) == 33

    # Input values
    _, x_bin_edges_list = corsika_histograms_instance.get_photon_radial_distr(bins=20, max_dist=10)
    for i_tel, _ in enumerate(corsika_histograms_instance.telescope_indices):
        assert np.amax(x_bin_edges_list[i_tel]) == 10
        assert np.size(x_bin_edges_list[i_tel]) == 21


def test_num_photons_per_event_per_telescope(corsika_histograms_instance_set_histograms):
    # Test number of photons in the first event
    assert np.shape(
        corsika_histograms_instance_set_histograms.num_photons_per_event_per_telescope
    ) == (
        87,
        2,
    )
    assert np.sum(
        corsika_histograms_instance_set_histograms.num_photons_per_event_per_telescope[:, 0]
    ) == pytest.approx(25425.8, abs=1e-1)
    # Test number of photons in the second event
    assert np.sum(
        corsika_histograms_instance_set_histograms.num_photons_per_event_per_telescope[:, 1]
    ) == pytest.approx(4582.9, abs=1e-1)

    # Decrease the number of telescopes and measure the number of photons on the ground again
    corsika_histograms_instance_set_histograms.set_histograms(telescope_indices=[3, 4, 5, 6])
    assert np.shape(
        corsika_histograms_instance_set_histograms.num_photons_per_event_per_telescope
    ) == (
        4,
        2,
    )
    assert np.sum(
        corsika_histograms_instance_set_histograms.num_photons_per_event_per_telescope[:, 0]
    ) == pytest.approx(7871.4, abs=1e-1)
    assert np.sum(
        corsika_histograms_instance_set_histograms.num_photons_per_event_per_telescope[:, 1]
    ) == pytest.approx(340.7, abs=1e-1)
    # Return the fixture to previous values
    corsika_histograms_instance_set_histograms.set_histograms()


def test_num_photons_per_event(corsika_histograms_instance_set_histograms):
    assert corsika_histograms_instance_set_histograms.num_photons_per_event[0] == pytest.approx(
        25425.8, abs=1e-1
    )
    assert corsika_histograms_instance_set_histograms.num_photons_per_event[1] == pytest.approx(
        4582.9, abs=1e-1
    )


def test_num_photons_per_telescope(corsika_histograms_instance_set_histograms):
    assert np.size(corsika_histograms_instance_set_histograms.num_photons_per_telescope) == 87
    assert np.sum(
        corsika_histograms_instance_set_histograms.num_photons_per_telescope
    ) == pytest.approx(25425.8 + 4582.9, abs=1e-1)


def test_get_num_photons_distr(corsika_histograms_instance_set_histograms, caplog):
    # Test range and bins for event
    hist, bin_edges = corsika_histograms_instance_set_histograms.get_num_photons_per_event_distr(
        bins=50, hist_range=None
    )
    assert np.size(bin_edges) == 51
    hist, bin_edges = corsika_histograms_instance_set_histograms.get_num_photons_per_event_distr(
        bins=100, hist_range=None
    )
    assert np.size(bin_edges) == 101
    hist, bin_edges = corsika_histograms_instance_set_histograms.get_num_photons_per_event_distr(
        bins=100, hist_range=(0, 500)
    )
    assert bin_edges[0][0] == 0
    assert bin_edges[0][-1] == 500

    # Test number of events simulated
    hist, bin_edges = corsika_histograms_instance_set_histograms.get_num_photons_per_event_distr(
        bins=2, hist_range=None
    )
    # Assert that the integration of the histogram resembles the known total number of events.
    assert np.sum(bin_edges[0, :-1] * hist[0]) / np.sum(
        corsika_histograms_instance_set_histograms.num_photons_per_event
    ) == pytest.approx(1, abs=1)

    # Test telescope
    (
        hist,
        bin_edges,
    ) = corsika_histograms_instance_set_histograms.get_num_photons_per_telescope_distr(
        bins=87, hist_range=None
    )
    # Assert that the integration of the histogram resembles the known total number of events.
    assert np.sum(bin_edges[0, :-1] * hist[0]) / np.sum(
        corsika_histograms_instance_set_histograms.num_photons_per_telescope
    ) == pytest.approx(1, abs=1)


def test_total_num_photons(corsika_histograms_instance_set_histograms):
    assert corsika_histograms_instance_set_histograms.total_num_photons == pytest.approx(
        30008.7, abs=1e-1
    )


def test_telescope_positions(corsika_histograms_instance_set_histograms):
    telescope_positions = corsika_histograms_instance_set_histograms.telescope_positions
    assert np.size(telescope_positions, axis=0) == 87
    coords_tel_0 = [-2064.0, -6482.0, 3400.0, 1250.0]

    for i_coord in range(4):
        assert telescope_positions[0][i_coord] == coords_tel_0[i_coord]

    # Test setting telescope position
    new_telescope_positions = copy.copy(telescope_positions)
    new_telescope_positions[0][0] = -400.0
    corsika_histograms_instance_set_histograms.telescope_positions = new_telescope_positions
    assert corsika_histograms_instance_set_histograms.telescope_positions[0][0] == -400.0


def test_event_zenith_angles(corsika_histograms_instance_set_histograms):
    for i_event in range(corsika_histograms_instance_set_histograms.num_events):
        assert corsika_histograms_instance_set_histograms.event_zenith_angles.value[i_event] == 20
    assert corsika_histograms_instance_set_histograms.event_zenith_angles.unit == u.deg


def test_event_azimuth_angles(corsika_histograms_instance_set_histograms):
    for i_event in range(corsika_histograms_instance_set_histograms.num_events):
        assert (
            np.around(corsika_histograms_instance_set_histograms.event_azimuth_angles.value)[
                i_event
            ]
            == -5
        )
    assert corsika_histograms_instance_set_histograms.event_azimuth_angles.unit == u.deg


def test_event_energies(corsika_histograms_instance_set_histograms):
    for i_event in range(corsika_histograms_instance_set_histograms.num_events):
        assert corsika_histograms_instance_set_histograms.event_energies.value[
            i_event
        ] == pytest.approx(0.01, abs=1e-2)
    assert corsika_histograms_instance_set_histograms.event_energies.unit == u.TeV


def test_event_first_interaction_heights(corsika_histograms_instance_set_histograms):
    first_height = [-10.3, -39.7]
    for i_event in range(corsika_histograms_instance_set_histograms.num_events):
        assert corsika_histograms_instance_set_histograms.event_first_interaction_heights.value[
            i_event
        ] == pytest.approx(first_height[i_event], abs=1e-1)
    assert corsika_histograms_instance_set_histograms.event_first_interaction_heights.unit == u.km


def test_magnetic_field(corsika_histograms_instance_set_histograms):
    for i_event in range(corsika_histograms_instance_set_histograms.num_events):
        assert corsika_histograms_instance_set_histograms.magnetic_field[0].value[
            i_event
        ] == pytest.approx(20.5, abs=1e-1)
        assert corsika_histograms_instance_set_histograms.magnetic_field[1].value[
            i_event
        ] == pytest.approx(-9.4, abs=1e-1)
    assert corsika_histograms_instance_set_histograms.magnetic_field[0].unit == u.uT


def test_get_event_parameter_info(corsika_histograms_instance_set_histograms, caplog):
    for parameter in corsika_histograms_instance_set_histograms.all_event_keys[1:]:
        assert isinstance(
            corsika_histograms_instance_set_histograms.get_event_parameter_info(parameter),
            u.quantity.Quantity,
        )
    with caplog.at_level("ERROR"):
        with pytest.raises(KeyError):
            corsika_histograms_instance_set_histograms.get_event_parameter_info(
                "non_existent_parameter"
            )
    assert (
        f"key is not valid. Valid entries are "
        f"{corsika_histograms_instance_set_histograms.all_event_keys}" in caplog.text
    )


def test_get_run_info(corsika_histograms_instance_set_histograms, caplog):
    for parameter in corsika_histograms_instance_set_histograms.all_run_keys[1:]:
        assert isinstance(
            corsika_histograms_instance_set_histograms.get_run_info(parameter),
            u.quantity.Quantity,
        )
    with caplog.at_level("ERROR"):
        with pytest.raises(KeyError):
            corsika_histograms_instance_set_histograms.get_run_info("non_existent_parameter")
    assert (
        f"key is not valid. Valid entries are "
        f"{corsika_histograms_instance_set_histograms.all_run_keys}" in caplog.text
    )


def test_event_1d_histogram(corsika_histograms_instance_set_histograms):
    hist, bin_edges = corsika_histograms_instance_set_histograms.event_1d_histogram(
        "total_energy", bins=5, hist_range=(5, 15)
    )
    assert np.size(bin_edges) == 6
    assert np.sum(hist) == 2
    assert hist[2] == 2


def test_event_2d_histogram(corsika_histograms_instance_set_histograms):
    hist, x_bin_edges, _ = corsika_histograms_instance_set_histograms.event_2d_histogram(
        "total_energy", "first_interaction_height", bins=(5, 5), hist_range=[[5, 15], [-60e5, -5e5]]
    )
    assert np.size(x_bin_edges) == 6
    assert np.sum(hist) == 2
    assert np.shape(hist) == (5, 5)


def test_get_bins_max_dist(corsika_histograms_instance):
    # Test when bins and max_dist are None
    bins, max_dist = corsika_histograms_instance._get_bins_max_dist()
    assert bins == 32  # half of the maximum bins
    assert max_dist == 16  # maximum stop value

    # Test when bins and max_dist are provided
    bins, max_dist = corsika_histograms_instance._get_bins_max_dist(bins=5, max_dist=15)
    assert bins == 5
    assert max_dist == 15

    # Test when only bins is provided
    bins, max_dist = corsika_histograms_instance._get_bins_max_dist(bins=7)
    assert bins == 7
    assert max_dist == 16  # maximum stop value

    # Test when only max_dist is provided
    bins, max_dist = corsika_histograms_instance._get_bins_max_dist(max_dist=12)
    assert bins == 32  # half of the maximum bins
    assert max_dist == 12


def test_meta_dict(corsika_histograms_instance_set_histograms):
    expected_meta_dict = {
        "corsika_version": corsika_histograms_instance_set_histograms.corsika_version,
        "simtools_version": version.__version__,
        "iact_file": corsika_histograms_instance_set_histograms.input_file.name,
        "telescope_indices": list(corsika_histograms_instance_set_histograms.telescope_indices),
        "individual_telescopes": corsika_histograms_instance_set_histograms.individual_telescopes,
        "note": "Only lower bin edges are given.",
    }
    assert corsika_histograms_instance_set_histograms._meta_dict == expected_meta_dict


def test_dict_1d_distributions(corsika_histograms_instance_set_histograms):
    expected_dict_1d_distributions = {
        "wavelength": {
            "function": "get_photon_wavelength_distr",
            "file name": "hist_1d_photon_wavelength_distr",
            "title": "Photon wavelength distribution",
            "bin edges": "wavelength",
            "axis unit": corsika_histograms_instance_set_histograms.hist_config["hist_position"][
                "z axis"
            ]["start"].unit,
        }
    }
    assert (
        corsika_histograms_instance_set_histograms.dict_1d_distributions["wavelength"]
        == expected_dict_1d_distributions["wavelength"]
    )


def test_export_and_read_histograms(corsika_histograms_instance_set_histograms, io_handler):
    # Default values
    corsika_histograms_instance_set_histograms.export_histograms()

    file_name = Path(corsika_histograms_instance_set_histograms.output_path).joinpath(
        "tel_output_10GeV-2-gamma-20deg-CTAO-South.hdf5"
    )
    assert io_handler.get_output_directory().joinpath(file_name).exists()

    # Change hdf5 file name
    corsika_histograms_instance_set_histograms.hdf5_file_name = "test.hdf5"
    corsika_histograms_instance_set_histograms.export_histograms()
    output_file = io_handler.get_output_directory().joinpath("test.hdf5")
    assert output_file.exists()

    # Read hdf5 file
    list_of_tables = read_hdf5(output_file)
    assert len(list_of_tables) == 12
    for table in list_of_tables:
        assert isinstance(table, Table)
    # Check piece of metadata
    assert (
        list_of_tables[-1].meta["corsika_version"]
        == corsika_histograms_instance_set_histograms.corsika_version
    )


def test_dict_2d_distributions(corsika_histograms_instance_set_histograms):
    expected_dict_2d_distributions = {
        "counts": {
            "function": "get_2d_photon_position_distr",
            "file name": "hist_2d_photon_count_distr",
            "title": "Photon count distribution on the ground",
            "x bin edges": "x position on the ground",
            "x axis unit": corsika_histograms_instance_set_histograms.hist_config["hist_position"][
                x_axis_string
            ]["start"].unit,
            "y bin edges": "y position on the ground",
            "y axis unit": corsika_histograms_instance_set_histograms.hist_config["hist_position"][
                y_axis_string
            ]["start"].unit,
        }
    }
    assert (
        corsika_histograms_instance_set_histograms.dict_2d_distributions["counts"]
        == expected_dict_2d_distributions["counts"]
    )


def test_export_event_header_1d_histogram(corsika_histograms_instance_set_histograms, io_handler):
    corsika_event_header_example = {
        "total_energy": "event_1d_histograms_total_energy",
        "azimuth": "event_1d_histograms_azimuth",
        "zenith": "event_1d_histograms_zenith",
        "first_interaction_height": "event_1d_histograms_first_interaction_height",
    }
    for event_header_element in corsika_event_header_example:
        corsika_histograms_instance_set_histograms.export_event_header_1d_histogram(
            event_header_element, bins=50, hist_range=None
        )

    tables = read_hdf5(corsika_histograms_instance_set_histograms.hdf5_file_name)
    assert len(tables) == 4


def test_export_event_header_2d_histogram(corsika_histograms_instance_set_histograms, io_handler):
    # Test writing the default photon histograms as well
    corsika_histograms_instance_set_histograms.export_histograms()
    tables = read_hdf5(corsika_histograms_instance_set_histograms.hdf5_file_name)
    assert len(tables) == 12

    corsika_event_header_example = {
        ("azimuth", "zenith"): "event_2d_histograms_azimuth_zenith",
    }

    # Test writing (appending) event header histograms
    for event_header_element, file_name in corsika_event_header_example.items():
        corsika_histograms_instance_set_histograms.export_event_header_2d_histogram(
            event_header_element[0], event_header_element[1], bins=50, hist_range=None
        )
    tables = read_hdf5(corsika_histograms_instance_set_histograms.hdf5_file_name)
    assert len(tables) == 13
