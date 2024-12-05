#!/usr/bin/python3

import pytest

import simtools.simtel.simtel_table_reader as simtel_table_reader


@pytest.fixture
def spe_test_file():
    return "tests/resources/spe_LST_2022-04-27_AP2.0e-4.dat"


@pytest.fixture
def spe_meta_test_comment():
    return "Norm_spe processing of single-p.e. response."


def test_read_simtel_table(spe_test_file, spe_meta_test_comment):
    """Test reading of sim_telarray table file into strings."""

    data, meta = simtel_table_reader._read_simtel_table(spe_test_file)

    assert isinstance(meta, str)
    assert spe_meta_test_comment in meta

    assert isinstance(data, str)
    assert len(data) > 0


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
    assert table.meta["name"] == parameter_name
    assert table.meta["file"] == spe_test_file
    assert spe_meta_test_comment in table.meta["context_from_sim_telarray"]
