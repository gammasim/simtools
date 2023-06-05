#!/usr/bin/python3

import pytest
from astropy import units as u

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
