#!/usr/bin/python3


import boost_histogram as bh
import numpy as np
import pytest
from astropy import units as u
from astropy.io.misc import yaml

import simtools.util.general as gen
from simtools.corsika.corsika_output import CorsikaOutput

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
    assert corsika_output_instance.telescope_indices == [0, 1, 2]
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

    # check log
    assert (
        "Finished reading the file and creating the histograms in "
        in corsika_output_instance._logger
    )


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
