#!/usr/bin/python3

from unittest import mock

import pytest

import simtools.simtel.simtel_table_reader as simtel_table_reader


@pytest.fixture
def spe_test_file():
    return "tests/resources/spe_LST_2022-04-27_AP2.0e-4.dat"


@pytest.fixture
def spe_meta_test_comment():
    return "Norm_spe processing of single-p.e. response."


def test_read_simtel_data(spe_test_file, spe_meta_test_comment):
    """Test reading of sim_telarray table file into strings."""

    data, meta, n_columns, n_dim = simtel_table_reader._read_simtel_data(spe_test_file)

    assert isinstance(meta, str)
    assert spe_meta_test_comment in meta

    assert isinstance(data, list)
    assert len(data) > 0

    assert n_dim is None
    assert n_columns == 3


@mock.patch("simtools.utils.general.read_file_encoded_in_utf_or_latin")
def test_read_simtel_data_rpol(mock_read_file_encoded_in_utf_or_latin):
    mock_read_file_encoded_in_utf_or_latin.return_value = [
        "#@RPOL@[ANGLE=] 2",
        "ANGLE=     0              10              20",
        "# Blabla",
        "200     0.9353535       0.9349322       0.9332631",
    ]
    rows, meta, n_columns, n_dim_axis = simtel_table_reader._read_simtel_data("test_file")
    assert n_columns == 4
    assert n_dim_axis == ["0", "10", "20"]
    assert "Blabla" in meta
    assert len(rows) == 1


def test_read_simtel_table_to_table(spe_test_file, spe_meta_test_comment):
    """Test reading of sim_telarray pm_photoelectron_spectrum table file into astropy table."""

    parameter_name = "pm_photoelectron_spectrum"

    table = simtel_table_reader.read_simtel_table(parameter_name, spe_test_file)

    assert len(table) == 2101
    assert "amplitude" in table.columns
    assert "response" in table.columns
    assert "response_with_ap" in table.columns
    assert table["amplitude"].unit is None
    assert table["response"].unit is None
    assert table["response_with_ap"].unit is None
    assert table.meta["Name"] == parameter_name
    assert table.meta["File"] == spe_test_file
    assert spe_meta_test_comment in table.meta["Context_from_sim_telarray"]

    with pytest.raises(
        ValueError, match="Unsupported parameter for sim_telarray table reading: not_a_parameter"
    ):
        simtel_table_reader.read_simtel_table("not_a_parameter", spe_test_file)


def test_data_simple_columns():
    columns = [
        "pm_photoelectron_spectrum",
        "quantum_efficiency",
        "camera_filter",
        "lightguide_efficiency_vs_wavelength",
        "lightguide_efficiency_vs_incidence_angle",
        "nsb_reference_spectrum",
    ]
    for column in columns:
        column_list, description = simtel_table_reader._data_columns(column, 2, None)
        assert isinstance(column_list, list)
        assert isinstance(description, str)


def test_data_columns_mirror_reflectivity():

    columns, description = simtel_table_reader._data_columns_mirror_reflectivity(2, ["0", "20"])
    assert columns == [
        {"name": "wavelength", "description": "Wavelength", "unit": "nm"},
        {"name": "reflectivity_0deg", "description": "Mirror reflectivity at 0 deg", "unit": None},
        {
            "name": "reflectivity_20deg",
            "description": "Mirror reflectivity at 20 deg",
            "unit": None,
        },
    ]
    assert description == "Mirror reflectivity"
    for n_columns in [2, 3, 4]:
        columns, _ = simtel_table_reader._data_columns_mirror_reflectivity(n_columns, None)
        assert len(columns) == n_columns


def test_data_columns_pulse_shape():
    columns, description = simtel_table_reader._data_columns_pulse_shape(2)
    assert columns == [
        {"name": "time", "description": "Time", "unit": "ns"},
        {"name": "amplitude", "description": "Amplitude", "unit": None},
    ]
    assert description == "Pulse shape"
    for n_columns in [2, 3]:
        columns, _ = simtel_table_reader._data_columns_pulse_shape(n_columns)
        assert len(columns) == n_columns


@mock.patch("simtools.simtel.simtel_table_reader._data_columns_mirror_reflectivity")
@mock.patch("simtools.simtel.simtel_table_reader._data_columns_pulse_shape")
def test_data_columns(mock_data_columns_mirror_reflectivity, mock_data_columns_pulse_shape):

    simtel_table_reader._data_columns("mirror_reflectivity", 2, ["0", "20"])
    assert mock_data_columns_mirror_reflectivity.called_once_with(2, ["0", "20"])

    simtel_table_reader._data_columns("fadc_pulse_shape", 4, None)
    assert mock_data_columns_pulse_shape.called_once_with(4)
