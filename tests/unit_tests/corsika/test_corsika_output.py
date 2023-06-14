#!/usr/bin/python3


import copy

import boost_histogram as bh
import numpy as np
import pytest
from astropy import units as u
from astropy.io.misc import yaml

import simtools.util.general as gen
from simtools.corsika.corsika_output import CorsikaOutput, HistogramNotCreated

test_file_name = "tel_output_10GeV-2-gamma-20deg-CTAO-South.dat"
# test_file_name = "tel_output.dat"


@pytest.fixture
def corsika_output_file(io_handler):
    corsika_output = gen.find_file(
        test_file_name,
        io_handler.get_output_directory(dir_type="corsika_output", test=True),
    )
    return corsika_output


@pytest.fixture
def corsika_output_instance(db, io_handler, corsika_output_file):
    # db.export_file_db(
    #    db_name="test-data",
    #    dest=io_handler.get_output_directory(dir_type="corsika_output", test=True),
    #    file_name=test_file_name,
    # )
    # return CorsikaOutput(corsika_output_file)
    return CorsikaOutput(test_file_name)


@pytest.fixture
def corsika_output_instance_set_histograms(db, io_handler, corsika_output_file):
    corsika_output_instance_to_set = CorsikaOutput(test_file_name)
    corsika_output_instance_to_set.set_histograms()
    return corsika_output_instance_to_set


def test_file_exists(corsika_output_file):
    assert corsika_output_file.exists()


def test_init(corsika_output_instance):
    assert corsika_output_instance.input_file.name == test_file_name
    with pytest.raises(FileNotFoundError):
        CorsikaOutput("wrong_file_name")
    assert len(corsika_output_instance.event_information) > 15
    assert "zenith" in corsika_output_instance.event_information
    assert len(corsika_output_instance.header) > 10

    # Check the some elements of the header
    manual_header = {
        "run_number": 1 * u.dimensionless_unscaled,
        "date": 230208 * u.dimensionless_unscaled,
        "version": 7.741 * u.dimensionless_unscaled,
        "n_observation_levels": 1 * u.dimensionless_unscaled,
        "observation_height": [214700.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] * u.cm,
        "energy_min": 10 * u.GeV,
    }
    for key in manual_header:
        assert (
            pytest.approx(corsika_output_instance.header[key].value, 3) == manual_header[key].value
        )

    assert corsika_output_instance.version == 7.741


def test_telescope_indices(corsika_output_instance):
    corsika_output_instance.telescope_indices = [0, 1, 2]
    assert (corsika_output_instance.telescope_indices == [0, 1, 2]).all()
    # Test float as input
    corsika_output_instance.telescope_indices = 1
    assert corsika_output_instance.telescope_indices == [1]
    # Test non-integer indices
    with pytest.raises(TypeError):
        corsika_output_instance.telescope_indices = [1.5, 2, 2.5]


def test_event_information(corsika_output_instance):
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


def test_hist_config_default_config(corsika_output_instance):
    hist_config = corsika_output_instance.hist_config
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


def test_hist_config_no_config_warning(corsika_output_instance, caplog):
    with caplog.at_level("WARNING"):
        hist_config = corsika_output_instance.hist_config
        assert (
            "No configuration was defined before. The default config is being created now."
            in caplog.text
        )
    assert hist_config == corsika_output_instance._create_histogram_default_config()


def test_hist_config_save_and_read_yml(corsika_output_instance, io_handler):
    # Test producing the yaml file
    corsika_output_instance.hist_config_to_yaml()
    output_file = io_handler.get_output_file(file_name="hist_config.yml", dir_type="corsika")
    assert output_file.exists()
    # Test reading the correct yaml file
    with open(output_file) as file:
        hist_config = yaml.load(file)
        assert hist_config == corsika_output_instance.hist_config


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


def test_fill_histograms_no_rotation():
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

    azimuth_angle = None
    zenith_angle = None

    corsika_output_instance_fill = CorsikaOutput(test_file_name)
    corsika_output_instance_fill.individual_telescopes = False
    corsika_output_instance_fill.telescope_indices = [0]

    corsika_output_instance_fill._create_histograms(individual_telescopes=False)

    # No count in the histogram before filling it
    assert np.count_nonzero(corsika_output_instance_fill.hist_direction[0].values()) == 0
    corsika_output_instance_fill._fill_histograms(photons, azimuth_angle, zenith_angle)
    # At least one count in the histogram after filling it
    assert np.count_nonzero(corsika_output_instance_fill.hist_direction[0].values()) > 0


def test_set_histograms_all_telescopes_1_histogram(corsika_output_instance):
    # all telescopes, but 1 histogram
    corsika_output_instance.set_histograms(telescope_indices=None, individual_telescopes=False)
    assert np.shape(corsika_output_instance.hist_position[0].values()) == (100, 100, 80)
    # assert that the histograms are filled
    assert np.count_nonzero(corsika_output_instance.hist_position[0][:, :, sum].view().T) == 162
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

    hist_non_zero_bins = [859, 933, 975]
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


def test_raise_if_no_histogram():
    corsika_output_instance_not_hist = CorsikaOutput(test_file_name)
    with pytest.raises(HistogramNotCreated):
        corsika_output_instance_not_hist._raise_if_no_histogram()
        assert (
            "The histograms were not created. Please, use `create_histograms` to create "
            "histograms from the CORSIKA output file." in corsika_output_instance_not_hist._logger
        )


def test_get_hist_2D_projection(corsika_output_instance):
    label = "hist_non_existent"
    with pytest.raises(ValueError):
        corsika_output_instance._get_hist_2D_projection(label)
        assert (
            f"label is not valid. Valid entries are "
            f"{corsika_output_instance._allowed_2D_labels}" in corsika_output_instance._logger
        )

    corsika_output_instance.set_histograms()
    hist_sums = [11633, 29.1, 11634, 11634]  # sum of photons are the approximately the same
    # (except for the density hist, which is divided by the area)
    for i_label, label in enumerate(["counts", "density", "direction", "time_altitude"]):
        x_edges, y_edges, hist_values = corsika_output_instance._get_hist_2D_projection(label)
        assert np.shape(x_edges) == (1, 101)
        assert np.shape(y_edges) == (1, 101)
        assert np.shape(hist_values) == (1, 100, 100)
        assert pytest.approx(np.sum(hist_values), 1) == hist_sums[i_label]

    # Repeat the test for less telescopes and see that less photons are counted in
    corsika_output_instance.set_histograms(telescope_indices=[0, 1, 2])
    hist_sums = [3677, 9.2, 3677, 3677]
    for i_label, label in enumerate(["counts", "density", "direction", "time_altitude"]):
        x_edges, y_edges, hist_values = corsika_output_instance._get_hist_2D_projection(label)
        assert pytest.approx(np.sum(hist_values), 1) == hist_sums[i_label]


def test_get_2D_photon_position_distr(corsika_output_instance_set_histograms):
    density = corsika_output_instance_set_histograms.get_2D_photon_position_distr(density=True)

    # Test the values of the histogram
    assert pytest.approx(np.sum(density[2]), 1) == 29
    counts = corsika_output_instance_set_histograms.get_2D_photon_position_distr(density=False)
    assert pytest.approx(np.sum(counts[2]), 1) == 11633

    # The edges should be the same
    assert (counts[0] == density[0]).all()
    assert (counts[1] == density[1]).all()


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
    corsika_output_instance_set_histograms.telescope_indices = [0, 4, 10]
    (
        num_events_array,
        telescope_indices_array,
        num_photons_per_event_per_telescope,
    ) = corsika_output_instance_set_histograms.get_2D_num_photons_distr()
    assert len(num_events_array) == 3  # number of events in this output file + 1 (edges of hist)
    assert (telescope_indices_array == [0, 1, 2, 3]).all()
    assert (
        pytest.approx(num_photons_per_event_per_telescope[0, 0], 1) == 2543.3
    )  # 1st tel, 1st event
    assert (
        pytest.approx(num_photons_per_event_per_telescope[0, 1], 1) == 290.5
    )  # 1st tel, 2nd event
    assert (
        pytest.approx(num_photons_per_event_per_telescope[1, 0], 1) == 1741.1
    )  # 2nd tel, 1st event
    assert pytest.approx(num_photons_per_event_per_telescope[1, 1], 1) == 85.9  # 2nd tel, 2nd event


def test_get_hist_1D_projection(corsika_output_instance_set_histograms):

    with pytest.raises(ValueError):
        corsika_output_instance_set_histograms._get_hist_1D_projection("label_not_valid")
        assert (
            f"label_not_valid is not valid. Valid entries are"
            f"{corsika_output_instance_set_histograms._allowed_1D_labels}"
            in corsika_output_instance_set_histograms._logger
        )

    labels = ["wavelength", "time", "altitude"]
    expected_shape_of_edges = [(1, 81), (1, 101), (1, 101)]
    expected_shape_of_values = [(1, 80), (1, 100), (1, 100)]
    expected_mean = [125.4, 116.3, 116.3]
    expected_std = [153.4, 378, 2, 312.0]
    for i_hist, hist_label in enumerate(labels):
        x_edges_list, hist_1D_list = corsika_output_instance_set_histograms._get_hist_1D_projection(
            hist_label
        )
        assert np.shape(x_edges_list) == expected_shape_of_edges[i_hist]
        assert np.shape(hist_1D_list) == expected_shape_of_values[i_hist]
        assert pytest.approx(np.mean(hist_1D_list), 1) == expected_mean[i_hist]
        assert pytest.approx(np.std(hist_1D_list), 1) == expected_std[i_hist]


def test_get_photon_altitude_distr(corsika_output_instance_set_histograms):
    for returned_variable in range(2):
        assert (
            corsika_output_instance_set_histograms._get_hist_1D_projection("altitude")[
                returned_variable
            ]
            == corsika_output_instance_set_histograms.get_photon_altitude_distr()[returned_variable]
        ).all()


def test_get_photon_time_distr(corsika_output_instance_set_histograms):
    for returned_variable in range(2):
        assert (
            corsika_output_instance_set_histograms._get_hist_1D_projection("time")[
                returned_variable
            ]
            == corsika_output_instance_set_histograms.get_photon_time_distr()[returned_variable]
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
    x_edges_list, hist_1D_list = corsika_output_instance_set_histograms.get_photon_radial_distr()
    for i_hist, _ in enumerate(corsika_output_instance_set_histograms.telescope_indices):
        assert np.amax(x_edges_list[i_hist]) == 16
        assert np.size(x_edges_list[i_hist]) == 33


def test_get_photon_radial_distr_some_telescopes(corsika_output_instance_set_histograms):

    # All given telescopes together
    corsika_output_instance_set_histograms.set_histograms(
        telescope_indices=[0, 1, 2, 3, 4, 5], individual_telescopes=False, hist_config=None
    )
    x_edges_list, hist_1D_list = corsika_output_instance_set_histograms.get_photon_radial_distr()
    assert np.amax(x_edges_list) == 1000
    assert np.size(x_edges_list) == 51


def test_get_photon_radial_distr_input_some_tel_and_density(corsika_output_instance_set_histograms):
    # Retrieve input values
    corsika_output_instance_set_histograms.set_histograms(
        telescope_indices=None, individual_telescopes=False, hist_config=None
    )

    x_edges_list, hist_1D_list = corsika_output_instance_set_histograms.get_photon_radial_distr(
        density=False, num_bins=100, max_dist=1200
    )
    assert np.amax(x_edges_list) == 1200
    assert np.size(x_edges_list) == 101

    # Test if the keyword density changes the output histogram but not the edges
    (
        x_edges_list_dens,
        hist_1D_list_dens,
    ) = corsika_output_instance_set_histograms.get_photon_radial_distr(
        density=True,
        num_bins=100,
        max_dist=1200,
    )
    assert (x_edges_list_dens == x_edges_list).all()

    print(np.sum(hist_1D_list_dens), np.sum(hist_1D_list))
    assert pytest.approx(np.sum(hist_1D_list_dens), 2) == 1.86  # density smaller because it divides
    # by the area (not counts per bin)
    assert pytest.approx(np.sum(hist_1D_list), 2) == 744.17


def test_get_photon_radial_distr_input_all_tel(corsika_output_instance):
    corsika_output_instance.set_histograms(
        telescope_indices=[0, 1, 2, 3, 4, 5], individual_telescopes=True
    )

    # Default input values
    x_edges_list, hist_1D_list = corsika_output_instance.get_photon_radial_distr()
    for i_tel, _ in enumerate(corsika_output_instance.telescope_indices):
        assert np.amax(x_edges_list[i_tel]) == 16
        assert np.size(x_edges_list[i_tel]) == 33

    # Input values
    x_edges_list, hist_1D_list = corsika_output_instance.get_photon_radial_distr(
        num_bins=20, max_dist=10
    )
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
            1,
        )
        == 25425.8
    )
    # Test number of photons in the second event
    assert (
        pytest.approx(
            np.sum(
                corsika_output_instance_set_histograms.num_photons_per_event_per_telescope[:, 1]
            ),
            1,
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
            1,
        )
        == 7871.4
    )
    assert (
        pytest.approx(
            np.sum(
                corsika_output_instance_set_histograms.num_photons_per_event_per_telescope[:, 1]
            ),
            1,
        )
        == 340.7
    )
    # Return the fixture to previous values
    corsika_output_instance_set_histograms.set_histograms()


def test_num_photons_per_event(corsika_output_instance_set_histograms):
    assert (
        pytest.approx(corsika_output_instance_set_histograms.num_photons_per_event[0], 1) == 25425.8
    )
    assert (
        pytest.approx(corsika_output_instance_set_histograms.num_photons_per_event[1], 1) == 4582.9
    )


def test_num_photons_per_telescope(corsika_output_instance_set_histograms):
    assert np.size(corsika_output_instance_set_histograms.num_photons_per_telescope) == 87
    assert (
        pytest.approx(np.sum(corsika_output_instance_set_histograms.num_photons_per_telescope), 1)
        == 25425.8 + 4582.9
    )


def test_get_num_photons_distr(corsika_output_instance_set_histograms, caplog):

    # Test range and bins for event
    edges, hist = corsika_output_instance_set_histograms.get_num_photons_distr(
        bins=50, range=None, event_or_telescope="event"
    )
    assert np.size(edges) == 51
    edges, hist = corsika_output_instance_set_histograms.get_num_photons_distr(
        bins=100, range=None, event_or_telescope="event"
    )
    assert np.size(edges) == 101
    edges, hist = corsika_output_instance_set_histograms.get_num_photons_distr(
        bins=100, range=(0, 500), event_or_telescope="event"
    )
    assert edges[0] == 0
    assert edges[-1] == 500

    # Test number of events simulated
    edges, hist = corsika_output_instance_set_histograms.get_num_photons_distr(
        bins=2, range=None, event_or_telescope="event"
    )
    # Assert that the integration of the histogram resembles the known total number of events.
    assert (
        pytest.approx(
            np.sum(edges[:-1] * hist)
            / np.sum(corsika_output_instance_set_histograms.num_photons_per_event),
            1,
        )
        == 1
    )

    # Test telescope
    edges, hist = corsika_output_instance_set_histograms.get_num_photons_distr(
        bins=87, range=None, event_or_telescope="telescope"
    )
    print(hist)
    # Assert that the integration of the histogram resembles the known total number of events.
    assert (
        pytest.approx(
            np.sum(edges[:-1] * hist)
            / np.sum(corsika_output_instance_set_histograms.num_photons_per_telescope),
            1,
        )
        == 1
    )

    with pytest.raises(ValueError):
        corsika_output_instance_set_histograms.get_num_photons_distr(
            bins=50, range=None, event_or_telescope="not_valid_name"
        )
        msg = "`event_or_telescope` has to be either 'event' or 'telescope'."
        msg in caplog.text


def test_total_num_photons(corsika_output_instance_set_histograms):
    assert pytest.approx(corsika_output_instance_set_histograms.total_num_photons, 1) == 30008.7


def test_telescope_positions(corsika_output_instance_set_histograms):
    telescope_positions = corsika_output_instance_set_histograms.telescope_positions
    assert np.size(telescope_positions, axis=0) == 87
    coords_tel_0 = [-2064.0, -6482.0, 3400.0, 1250.0]

    for i_coord in range(4):
        assert telescope_positions[0][i_coord] == coords_tel_0[i_coord]
