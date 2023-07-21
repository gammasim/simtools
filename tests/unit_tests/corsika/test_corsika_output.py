#!/usr/bin/python3


import copy
from pathlib import Path

import boost_histogram as bh
import numpy as np
import pytest
from astropy import units as u

from simtools.corsika.corsika_output import CorsikaOutput, HistogramNotCreated


def test_init(corsika_output_instance, corsika_output_file_name):
    assert corsika_output_instance.input_file.name == Path(corsika_output_file_name).name
    with pytest.raises(FileNotFoundError):
        CorsikaOutput("wrong_file_name")
    assert len(corsika_output_instance.event_information) > 15
    assert "zenith" in corsika_output_instance.event_information


def test_version(corsika_output_instance):
    assert corsika_output_instance.version == 7.741


def test_initialize_header(corsika_output_instance):
    corsika_output_instance._initialize_header()
    # Check the some elements of the header
    manual_header = {
        "run_number": 1 * u.dimensionless_unscaled,
        "date": 230208 * u.dimensionless_unscaled,
        "version": 7.741 * u.dimensionless_unscaled,
        "n_observation_levels": 1 * u.dimensionless_unscaled,
        "observation_height": [214700.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] * u.cm,
        "energy_min": 10 * u.GeV,
    }
    assert len(corsika_output_instance.header) > 10
    for key in manual_header:
        assert pytest.approx(corsika_output_instance.header[key].value) == manual_header[key].value


def test_telescope_indices(corsika_output_instance):
    corsika_output_instance.telescope_indices = [0, 1, 2]
    assert (corsika_output_instance.telescope_indices == [0, 1, 2]).all()
    # Test int as input
    corsika_output_instance.telescope_indices = 1
    assert corsika_output_instance.telescope_indices == [1]
    # Test non-integer indices
    with pytest.raises(TypeError):
        corsika_output_instance.telescope_indices = [1.5, 2, 2.5]


def test_read_event_information(corsika_output_instance):
    # Check some of the event info
    manual_event_info = {
        "event_number": [1, 2] * u.dimensionless_unscaled,
        "particle_id": [1, 1] * u.dimensionless_unscaled,
        "total_energy": [10, 10] * u.GeV,
        "starting_altitude": [0, 0] * (u.g / (u.cm**2)),
    }
    for key in manual_event_info:
        assert (corsika_output_instance.event_information[key] == manual_event_info[key]).all()


def test_get_header_astropy_units(corsika_output_instance):
    parameters = ["momentum_x", "event_number", "starting_altitude", "total_energy"]
    non_astropy_units = ["GeV/c", "", "g/cm2", "GeV"]
    all_event_astropy_units = corsika_output_instance._get_header_astropy_units(
        parameters, non_astropy_units
    )
    for astropy_unit in all_event_astropy_units.values():
        assert isinstance(astropy_unit, u.core.CompositeUnit)


def test_hist_config_default_config(corsika_output_instance, caplog):
    with caplog.at_level("WARNING"):
        hist_config = corsika_output_instance.hist_config
        assert "No histogram configuration was defined before." in caplog.text
    assert isinstance(hist_config, dict)
    assert hist_config == corsika_output_instance._create_histogram_default_config()


def test_hist_config_custom_config(corsika_output_instance):
    custom_config = {
        "hist_position": {
            "x axis": {"bins": 100, "start": -1000, "stop": 1000, "scale": "linear"},
            "y axis": {"bins": 100, "start": -1000, "stop": 1000, "scale": "linear"},
            "z axis": {"bins": 80, "start": 200, "stop": 1000, "scale": "linear"},
        },
        "hist_direction": {
            "azimuth": {"bins": 36, "start": 0, "stop": 360, "scale": "linear"},
            "zenith": {"bins": 18, "start": 0, "stop": 90, "scale": "linear"},
        },
    }
    corsika_output_instance._hist_config = custom_config
    hist_config = corsika_output_instance.hist_config
    assert hist_config == custom_config


def test_hist_config_save_and_read_yml(corsika_output_instance, io_handler):
    # Test producing the yaml file
    temp_hist_config = corsika_output_instance.hist_config
    corsika_output_instance.hist_config_to_yaml()
    output_file = io_handler.get_output_file(file_name="hist_config.yml", dir_type="corsika")
    corsika_output_instance.hist_config = output_file
    assert corsika_output_instance.hist_config == temp_hist_config


def test_create_regular_axes_valid_label(corsika_output_instance):
    hists = ["hist_position", "hist_direction", "hist_time_altitude"]
    num_of_axes = [3, 2, 2]
    for i_hist, _ in enumerate(hists):
        axes = corsika_output_instance._create_regular_axes(hists[i_hist])
        assert len(axes) == num_of_axes[i_hist]
        for i_axis in range(num_of_axes[i_hist]):
            assert isinstance(axes[i_axis], bh.axis.Regular)


def test_create_regular_axes_invalid_label(corsika_output_instance):
    label = "invalid_label"
    with pytest.raises(ValueError):
        corsika_output_instance._create_regular_axes(label)


def test_create_histograms(corsika_output_instance):
    # Test once for individual_telescopes True and once false
    individual_telescopes = [True, False]
    corsika_output_instance.telescope_indices = [0, 1, 2, 3]
    num_of_expected_hists = [4, 1]
    for i_test in range(2):
        corsika_output_instance._create_histograms(individual_telescopes[i_test])
        assert corsika_output_instance.num_of_hist == num_of_expected_hists[i_test]
        assert len(corsika_output_instance.hist_position) == num_of_expected_hists[i_test]
        assert len(corsika_output_instance.hist_direction) == num_of_expected_hists[i_test]
        assert len(corsika_output_instance.hist_time_altitude) == num_of_expected_hists[i_test]
        assert isinstance(corsika_output_instance.hist_position[0], bh.Histogram)
        assert isinstance(corsika_output_instance.hist_direction[0], bh.Histogram)
        assert isinstance(corsika_output_instance.hist_time_altitude[0], bh.Histogram)


def test_fill_histograms_no_rotation(corsika_output_file_name):
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

    corsika_output_instance_fill = CorsikaOutput(corsika_output_file_name)
    corsika_output_instance_fill.individual_telescopes = False
    corsika_output_instance_fill.telescope_indices = [0]

    corsika_output_instance_fill._create_histograms(individual_telescopes=False)

    # No count in the histogram before filling it
    assert np.count_nonzero(corsika_output_instance_fill.hist_direction[0].values()) == 0
    corsika_output_instance_fill._fill_histograms(
        photons, rotation_around_z_axis=None, rotation_around_y_axis=None
    )
    # At least one count in the histogram after filling it
    assert np.count_nonzero(corsika_output_instance_fill.hist_direction[0].values()) > 0


def test_get_hist_1D_projection(corsika_output_instance_set_histograms, caplog):

    with pytest.raises(ValueError):
        corsika_output_instance_set_histograms._get_hist_1D_projection("label_not_valid")
        assert "label_not_valid is not valid." in caplog.text

    labels = ["wavelength", "time", "altitude"]
    expected_shape_of_edges = [(1, 81), (1, 101), (1, 101)]
    expected_shape_of_values = [(1, 80), (1, 100), (1, 100)]
    expected_mean = [125.4, 116.3, 116.3]
    expected_std = [153.4, 378.2, 483.8]
    for i_hist, hist_label in enumerate(labels):
        hist_1D_list, x_edges_list = corsika_output_instance_set_histograms._get_hist_1D_projection(
            hist_label
        )
        assert np.shape(x_edges_list) == expected_shape_of_edges[i_hist]
        assert np.shape(hist_1D_list) == expected_shape_of_values[i_hist]
        assert pytest.approx(np.mean(hist_1D_list), 1e-2) == expected_mean[i_hist]
        assert pytest.approx(np.std(hist_1D_list), 1e-2) == expected_std[i_hist]


def test_set_histograms_all_telescopes_1_histogram(corsika_output_instance):
    # all telescopes, but 1 histogram
    corsika_output_instance.set_histograms(telescope_indices=None, individual_telescopes=False)
    assert np.shape(corsika_output_instance.hist_position[0].values()) == (100, 100, 80)
    # assert that the histograms are filled
    assert np.count_nonzero(corsika_output_instance.hist_position[0][:, :, sum].view().T) == 159
    # and the sum is what we expect
    assert np.sum(corsika_output_instance.hist_position[0].view()) == 10031.0


def test_set_histograms_3_telescopes_1_histogram(corsika_output_instance):
    # 3 telescopes, but 1 histogram
    corsika_output_instance.set_histograms(telescope_indices=[0, 1, 2], individual_telescopes=False)
    assert np.shape(corsika_output_instance.hist_position[0].values()) == (100, 100, 80)
    # assert that the histograms are filled
    assert np.count_nonzero(corsika_output_instance.hist_position[0][:, :, sum].view().T) == 12
    # and the sum is what we expect
    assert np.sum(corsika_output_instance.hist_position[0].view()) == 3177.0


def test_set_histograms_3_telescopes_3_histograms(corsika_output_instance):
    # 3 telescopes and 3 histograms
    corsika_output_instance.set_histograms(telescope_indices=[0, 1, 2], individual_telescopes=True)

    hist_non_zero_bins = [827, 911, 966]
    hist_sum = [959.0, 1062.0, 1156.0]
    for i_hist in range(3):
        assert np.shape(corsika_output_instance.hist_position[i_hist].values()) == (64, 64, 80)
        # assert that the histograms are filled
        assert (
            np.count_nonzero(corsika_output_instance.hist_position[i_hist][:, :, sum].view().T)
            == hist_non_zero_bins[i_hist]
        )
        # and the sum is what we expect
        assert np.sum(corsika_output_instance.hist_position[i_hist].view()) == hist_sum[i_hist]


def test_set_histograms_passing_config(corsika_output_instance):
    new_hist_config = copy.copy(corsika_output_instance.hist_config)
    xy_maximum = 500 * u.m
    xy_bin = 100
    new_hist_config["hist_position"] = {
        "x axis": {
            "bins": xy_bin,
            "start": -xy_maximum,
            "stop": xy_maximum,
            "scale": "linear",
        },
        "y axis": {
            "bins": xy_bin,
            "start": -xy_maximum,
            "stop": xy_maximum,
            "scale": "linear",
        },
        "z axis": {
            "bins": 80,
            "start": 200 * u.nm,
            "stop": 1000 * u.nm,
            "scale": "linear",
        },
    }
    corsika_output_instance.set_histograms(individual_telescopes=False, hist_config=new_hist_config)
    assert corsika_output_instance.hist_position[0][:, :, sum].shape == (100, 100)
    assert corsika_output_instance.hist_position[0][:, :, sum].axes[0].edges[0] == -500
    assert corsika_output_instance.hist_position[0][:, :, sum].axes[0].edges[-1] == 500


def test_raise_if_no_histogram(corsika_output_file_name, caplog):
    corsika_output_instance_not_hist = CorsikaOutput(corsika_output_file_name)
    with pytest.raises(HistogramNotCreated):
        corsika_output_instance_not_hist._raise_if_no_histogram()
        assert "The histograms were not created." in caplog


def test_get_hist_2D_projection(corsika_output_instance, caplog):

    corsika_output_instance.set_histograms()

    label = "hist_non_existent"
    with pytest.raises(ValueError):
        corsika_output_instance._get_hist_2D_projection(label)
        assert "label is not valid." in caplog.text

    labels = ["counts", "density", "direction", "time_altitude"]
    hist_sums = [11633, 29.1, 11634, 11634]  # sum of photons are approximately the same
    # (except for the density hist, which is divided by the area)
    for i_label, label in enumerate(labels):
        hist_values, x_edges, y_edges = corsika_output_instance._get_hist_2D_projection(label)
        assert np.shape(x_edges) == (1, 101)
        assert np.shape(y_edges) == (1, 101)
        assert np.shape(hist_values) == (1, 100, 100)
        assert pytest.approx(np.sum(hist_values), 1e-2) == hist_sums[i_label]

    # Repeat the test for fewer telescopes and see that less photons are counted in
    corsika_output_instance.set_histograms(telescope_indices=[0, 1, 2])
    hist_sums = [3677, 9.2, 3677, 3677]
    for i_label, label in enumerate(labels):
        hist_values, x_edges, y_edges = corsika_output_instance._get_hist_2D_projection(label)
        assert pytest.approx(np.sum(hist_values), 1e-2) == hist_sums[i_label]


def test_get_2D_photon_position_distr(corsika_output_instance_set_histograms):
    density = corsika_output_instance_set_histograms.get_2D_photon_density_distr()

    # Test the values of the histogram
    assert pytest.approx(np.sum(density[0]), 1e-2) == 29
    counts = corsika_output_instance_set_histograms.get_2D_photon_position_distr()
    assert pytest.approx(np.sum(counts[0]), 1e-2) == 11633

    # The edges should be the same
    assert (counts[1] == density[1]).all()
    assert (counts[2] == density[2]).all()


def test_get_2D_photon_direction_distr(corsika_output_instance_set_histograms):
    for returned_variable in range(3):
        assert (
            corsika_output_instance_set_histograms.get_2D_photon_direction_distr()[
                returned_variable
            ]
            == corsika_output_instance_set_histograms._get_hist_2D_projection("direction")[
                returned_variable
            ]
        ).all()


def test_get_2D_photon_time_altitude(corsika_output_instance_set_histograms):
    for returned_variable in range(3):
        assert (
            corsika_output_instance_set_histograms.get_2D_photon_time_altitude()[returned_variable]
            == corsika_output_instance_set_histograms._get_hist_2D_projection("time_altitude")[
                returned_variable
            ]
        ).all()


def test_get_2D_num_photons_distr(corsika_output_instance_set_histograms):
    corsika_output_instance_set_histograms.set_histograms(telescope_indices=[0, 4, 10])
    num_photons_per_event_per_telescope, num_events_array, telescope_indices_array = corsika_output_instance_set_histograms.get_2D_num_photons_distr()
    assert np.shape(num_events_array) == (1,3)  # number of events in this output file + 1 (edges of hist)
    assert (telescope_indices_array == [0, 1, 2, 3]).all()
    print(np.shape(num_photons_per_event_per_telescope))
    assert (
        pytest.approx(num_photons_per_event_per_telescope[0][0, 0], 1e-2) == 2543.3
    )  # 1st tel, 1st event
    assert (
        pytest.approx(num_photons_per_event_per_telescope[0][0, 1], 1e-2) == 290.4
    )  # 1st tel, 2nd event
    assert (
        pytest.approx(num_photons_per_event_per_telescope[0][1, 0], 1e-2) == 1741
    )  # 2nd tel, 1st event
    assert (
        pytest.approx(num_photons_per_event_per_telescope[0][1, 1], 1e-2) == 85.9
    )  # 2nd tel, 2nd event


def test_get_photon_altitude_distr(corsika_output_instance_set_histograms):
    for returned_variable in range(2):
        assert (
            corsika_output_instance_set_histograms._get_hist_1D_projection("altitude")[
                returned_variable
            ]
            == corsika_output_instance_set_histograms.get_photon_altitude_distr()[returned_variable]
        ).all()


def test_get_photon_time_of_emission_distr(corsika_output_instance_set_histograms):
    for returned_variable in range(2):
        assert (
            corsika_output_instance_set_histograms._get_hist_1D_projection("time")[
                returned_variable
            ]
            == corsika_output_instance_set_histograms.get_photon_time_of_emission_distr()[
                returned_variable
            ]
        ).all()


def test_get_photon_wavelength_distr(corsika_output_instance_set_histograms):
    for returned_variable in range(2):
        assert (
            corsika_output_instance_set_histograms._get_hist_1D_projection("wavelength")[
                returned_variable
            ]
            == corsika_output_instance_set_histograms.get_photon_wavelength_distr()[
                returned_variable
            ]
        ).all()


def test_get_photon_radial_distr_individual_telescopes(corsika_output_instance_set_histograms):

    # Individual telescopes
    corsika_output_instance_set_histograms.set_histograms(
        telescope_indices=[0, 1, 2], individual_telescopes=True, hist_config=None
    )
    _, x_edges_list = corsika_output_instance_set_histograms.get_photon_radial_distr()
    for i_hist, _ in enumerate(corsika_output_instance_set_histograms.telescope_indices):
        assert np.amax(x_edges_list[i_hist]) == 16
        assert np.size(x_edges_list[i_hist]) == 33


def test_get_photon_radial_distr_some_telescopes(corsika_output_instance_set_histograms):

    # All given telescopes together
    corsika_output_instance_set_histograms.set_histograms(
        telescope_indices=[0, 1, 2, 3, 4, 5], individual_telescopes=False, hist_config=None
    )
    _, x_edges_list = corsika_output_instance_set_histograms.get_photon_radial_distr()
    assert np.amax(x_edges_list) == 1000
    assert np.size(x_edges_list) == 51


def test_get_photon_radial_distr_input_some_tel_and_density(corsika_output_instance_set_histograms):
    # Retrieve input values
    corsika_output_instance_set_histograms.set_histograms(
        telescope_indices=None, individual_telescopes=False, hist_config=None
    )

    hist_1D_list, x_edges_list = corsika_output_instance_set_histograms.get_photon_radial_distr(bins=100, max_dist=1200)
    assert np.amax(x_edges_list) == 1200
    assert np.size(x_edges_list) == 101

    # Test if the keyword density changes the output histogram but not the edges
    (
        hist_1D_list_dens,
        x_edges_list_dens,
    ) = corsika_output_instance_set_histograms.get_photon_density_distr(
        bins=100,
        max_dist=1200,
    )
    assert (x_edges_list_dens == x_edges_list).all()

    assert (
        pytest.approx(np.sum(hist_1D_list_dens), 1e-2) == 1.86
    )  # density smaller because it divides
    # by the area (not counts per bin)
    assert pytest.approx(np.sum(hist_1D_list), 1e-2) == 744.17


def test_get_photon_radial_distr_input_all_tel(corsika_output_instance):
    corsika_output_instance.set_histograms(
        telescope_indices=[0, 1, 2, 3, 4, 5], individual_telescopes=True
    )

    # Default input values
    _, x_edges_list = corsika_output_instance.get_photon_radial_distr()
    for i_tel, _ in enumerate(corsika_output_instance.telescope_indices):
        assert np.amax(x_edges_list[i_tel]) == 16
        assert np.size(x_edges_list[i_tel]) == 33

    # Input values
    _, x_edges_list = corsika_output_instance.get_photon_radial_distr(bins=20, max_dist=10)
    for i_tel, _ in enumerate(corsika_output_instance.telescope_indices):
        assert np.amax(x_edges_list[i_tel]) == 10
        assert np.size(x_edges_list[i_tel]) == 21


def test_num_photons_per_event_per_telescope(corsika_output_instance_set_histograms):

    # Test number of photons in the first event
    assert np.shape(corsika_output_instance_set_histograms.num_photons_per_event_per_telescope) == (
        87,
        2,
    )
    assert (
        pytest.approx(
            np.sum(
                corsika_output_instance_set_histograms.num_photons_per_event_per_telescope[:, 0]
            ),
            1e-2,
        )
        == 25425.8
    )
    # Test number of photons in the second event
    assert (
        pytest.approx(
            np.sum(
                corsika_output_instance_set_histograms.num_photons_per_event_per_telescope[:, 1]
            ),
            1e-2,
        )
        == 4582.9
    )

    # Decrease the number of telescopes and measure the number of photons on the ground again
    corsika_output_instance_set_histograms.set_histograms(telescope_indices=[3, 4, 5, 6])
    assert np.shape(corsika_output_instance_set_histograms.num_photons_per_event_per_telescope) == (
        4,
        2,
    )
    assert (
        pytest.approx(
            np.sum(
                corsika_output_instance_set_histograms.num_photons_per_event_per_telescope[:, 0]
            ),
            1e-2,
        )
        == 7871.4
    )
    assert (
        pytest.approx(
            np.sum(
                corsika_output_instance_set_histograms.num_photons_per_event_per_telescope[:, 1]
            ),
            1e-2,
        )
        == 340.7
    )
    # Return the fixture to previous values
    corsika_output_instance_set_histograms.set_histograms()


def test_num_photons_per_event(corsika_output_instance_set_histograms):
    assert (
        pytest.approx(corsika_output_instance_set_histograms.num_photons_per_event[0], 1e-2)
        == 25425.8
    )
    assert (
        pytest.approx(corsika_output_instance_set_histograms.num_photons_per_event[1], 1e-2)
        == 4582.9
    )


def test_num_photons_per_telescope(corsika_output_instance_set_histograms):
    assert np.size(corsika_output_instance_set_histograms.num_photons_per_telescope) == 87
    assert (
        pytest.approx(
            np.sum(corsika_output_instance_set_histograms.num_photons_per_telescope), 1e-2
        )
        == 25425.8 + 4582.9
    )


def test_get_num_photons_distr(corsika_output_instance_set_histograms, caplog):

    # Test range and bins for event
    hist, edges = corsika_output_instance_set_histograms.get_num_photons_per_event_distr(
        bins=50, hist_range=None
    )
    assert np.size(edges) == 51
    hist, edges = corsika_output_instance_set_histograms.get_num_photons_per_event_distr(
        bins=100, hist_range=None
    )
    assert np.size(edges) == 101
    hist, edges = corsika_output_instance_set_histograms.get_num_photons_per_event_distr(
        bins=100, hist_range=(0, 500)
    )
    assert edges[0][0] == 0
    assert edges[0][-1] == 500

    # Test number of events simulated
    hist, edges = corsika_output_instance_set_histograms.get_num_photons_per_event_distr(
        bins=2, hist_range=None
    )
    # Assert that the integration of the histogram resembles the known total number of events.
    print(np.sum(corsika_output_instance_set_histograms.num_photons_per_event))
    assert (
        pytest.approx(
            np.sum(edges[0,:-1] * hist[0])
            / np.sum(corsika_output_instance_set_histograms.num_photons_per_event),
            abs=1,
        )
        == 1
    )

    # Test telescope
    hist, edges = corsika_output_instance_set_histograms.get_num_photons_per_telescope_distr(
        bins=87, hist_range=None
    )
    # Assert that the integration of the histogram resembles the known total number of events.
    assert (
        pytest.approx(
            np.sum(edges[0,:-1] * hist[0])
            / np.sum(corsika_output_instance_set_histograms.num_photons_per_telescope),
            abs=1,
        )
        == 1
    )


def test_total_num_photons(corsika_output_instance_set_histograms):
    assert pytest.approx(corsika_output_instance_set_histograms.total_num_photons, 1e-2) == 30008.7


def test_telescope_positions(corsika_output_instance_set_histograms):
    telescope_positions = corsika_output_instance_set_histograms.telescope_positions
    assert np.size(telescope_positions, axis=0) == 87
    coords_tel_0 = [-2064.0, -6482.0, 3400.0, 1250.0]

    for i_coord in range(4):
        assert telescope_positions[0][i_coord] == coords_tel_0[i_coord]

    # Test setting telescope position
    new_telescope_positions = copy.copy(telescope_positions)
    new_telescope_positions[0][0] = -400.0
    corsika_output_instance_set_histograms.telescope_positions = new_telescope_positions
    assert corsika_output_instance_set_histograms.telescope_positions[0][0] == -400.0


def test_event_zenith_angles(corsika_output_instance_set_histograms):
    for i_event in range(corsika_output_instance_set_histograms.num_events):
        assert corsika_output_instance_set_histograms.event_zenith_angles.value[i_event] == 20
    assert corsika_output_instance_set_histograms.event_zenith_angles.unit == u.deg


def test_event_azimuth_angles(corsika_output_instance_set_histograms):
    for i_event in range(corsika_output_instance_set_histograms.num_events):
        assert (
            np.around(corsika_output_instance_set_histograms.event_azimuth_angles.value)[i_event]
            == -5
        )
    assert corsika_output_instance_set_histograms.event_azimuth_angles.unit == u.deg


def test_event_energies(corsika_output_instance_set_histograms):
    for i_event in range(corsika_output_instance_set_histograms.num_events):
        assert (
            pytest.approx(
                corsika_output_instance_set_histograms.event_energies.value[i_event], 1e-2
            )
            == 0.01
        )
    assert corsika_output_instance_set_histograms.event_energies.unit == u.TeV


def test_event_first_interaction_heights(corsika_output_instance_set_histograms):
    first_height = [-10.3, -39.7]
    for i_event in range(corsika_output_instance_set_histograms.num_events):
        assert (
            pytest.approx(
                corsika_output_instance_set_histograms.event_first_interaction_heights.value[
                    i_event
                ],
                1e-2,
            )
            == first_height[i_event]
        )
    assert corsika_output_instance_set_histograms.event_first_interaction_heights.unit == u.km


def test_magnetic_field(corsika_output_instance_set_histograms):
    for i_event in range(corsika_output_instance_set_histograms.num_events):
        assert (
            pytest.approx(
                corsika_output_instance_set_histograms.magnetic_field[0].value[i_event], 1e-2
            )
            == 20.5
        )
        assert (
            pytest.approx(
                corsika_output_instance_set_histograms.magnetic_field[1].value[i_event], 1e-2
            )
            == -9.4
        )
    assert corsika_output_instance_set_histograms.magnetic_field[0].unit == u.uT


def test_get_event_parameter_info(corsika_output_instance_set_histograms, caplog):
    for parameter in corsika_output_instance_set_histograms.all_event_keys[1:]:
        assert isinstance(
            corsika_output_instance_set_histograms.get_event_parameter_info(parameter),
            u.quantity.Quantity,
        )

    with pytest.raises(KeyError):
        corsika_output_instance_set_histograms.get_event_parameter_info("non_existent_parameter")
        assert (
            f"`key` is not valid. Valid entries are "
            f"{corsika_output_instance_set_histograms.all_event_keys}" in caplog.text
        )


def test_get_run_info(corsika_output_instance_set_histograms, caplog):
    for parameter in corsika_output_instance_set_histograms.all_run_keys[1:]:
        print(corsika_output_instance_set_histograms.get_run_info(parameter))
        assert isinstance(
            corsika_output_instance_set_histograms.get_run_info(parameter),
            u.quantity.Quantity,
        )

    with pytest.raises(KeyError):
        corsika_output_instance_set_histograms.get_run_info("non_existent_parameter")
        assert (
            f"`key` is not valid. Valid entries are "
            f"{corsika_output_instance_set_histograms.all_run_keys}" in caplog.text
        )


def test_event_1D_histogram(corsika_output_instance_set_histograms):
    hist, edges = corsika_output_instance_set_histograms.event_1D_histogram(
        "total_energy", bins=5, hist_range=(5, 15)
    )
    assert np.size(edges) == 6
    assert np.sum(hist) == 2
    assert hist[2] == 2


def test_event_2D_histogram(corsika_output_instance_set_histograms):
    hist, x_edges, _ = corsika_output_instance_set_histograms.event_2D_histogram(
        "total_energy", "first_interaction_height", bins=(5, 5), hist_range=[[5, 15], [-60e5, -5e5]]
    )
    assert np.size(x_edges) == 6
    assert np.sum(hist) == 2
    assert np.shape(hist) == (5, 5)
