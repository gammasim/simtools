#!/usr/bin/python3

"""Unit tests for illuminator_visibility module."""

import logging

import pytest

from simtools.model.illuminator_visibility import IlluminatorTelescopeVisibility


@pytest.fixture
def simple_visibility_data():
    """Create a simple test illuminator-telescope visibility dict."""
    return {
        "columns": ["illuminator_id", "telescope_id", "visible"],
        "rows": [
            ["ILLS-01", "MSTS-01", True],
            ["ILLS-01", "MSTS-02", False],
            ["ILLS-01", "MSTS-03", True],
            ["ILLS-02", "MSTS-01", False],
            ["ILLS-02", "MSTS-02", True],
            ["ILLS-02", "MSTS-03", True],
        ],
    }


@pytest.fixture
def north_visibility_data():
    """Create a visibility dict matching North site structure."""
    return {
        "columns": ["illuminator_id", "telescope_id", "visible"],
        "rows": [
            ["ILLN-01", "LSTN-01", False],
            ["ILLN-01", "LSTN-02", False],
            ["ILLN-01", "MSTN-01", True],
            ["ILLN-01", "MSTN-02", True],
            ["ILLN-02", "LSTN-01", True],
            ["ILLN-02", "LSTN-02", True],
            ["ILLN-02", "MSTN-01", True],
            ["ILLN-02", "MSTN-02", True],
        ],
    }


def test_init_valid_data(simple_visibility_data):
    """Test initialization with valid visibility data."""
    visibility = IlluminatorTelescopeVisibility(simple_visibility_data)
    assert visibility is not None
    assert visibility.n_illuminators == 2
    assert visibility.n_telescopes == 3


def test_init_invalid_type():
    """Test initialization with non-dict input."""
    with pytest.raises(ValueError, match="Expected dict"):
        IlluminatorTelescopeVisibility("not a dict")


def test_init_missing_keys():
    """Test initialization with missing required keys."""
    with pytest.raises(ValueError, match="must contain 'columns' and 'rows'"):
        IlluminatorTelescopeVisibility({"columns": ["a", "b", "c"]})


def test_init_missing_required_columns():
    """Test initialization when required column names are missing."""
    data = {
        "columns": ["wrong_column", "telescope_id", "visible"],
        "rows": [["ILL-01", "TEL-01", True]],
    }
    with pytest.raises(ValueError, match="must have columns 'illuminator_id'"):
        IlluminatorTelescopeVisibility(data)


def test_init_empty_rows():
    """Test initialization with no rows."""
    data = {
        "columns": ["illuminator_id", "telescope_id", "visible"],
        "rows": [],
    }
    with pytest.raises(ValueError, match="contains no illuminators"):
        IlluminatorTelescopeVisibility(data)


def test_get_illuminators(simple_visibility_data):
    """Test getting list of illuminators."""
    visibility = IlluminatorTelescopeVisibility(simple_visibility_data)
    illuminators = visibility.get_illuminators()

    assert len(illuminators) == 2
    assert "ILLS-01" in illuminators
    assert "ILLS-02" in illuminators


def test_get_telescopes(simple_visibility_data):
    """Test getting list of telescopes."""
    visibility = IlluminatorTelescopeVisibility(simple_visibility_data)
    telescopes = visibility.get_telescopes()

    assert len(telescopes) == 3
    assert "MSTS-01" in telescopes
    assert "MSTS-02" in telescopes
    assert "MSTS-03" in telescopes


def test_get_valid_pairs(simple_visibility_data):
    """Test getting all valid illuminator-telescope pairs."""
    visibility = IlluminatorTelescopeVisibility(simple_visibility_data)
    pairs = visibility.get_valid_pairs()

    assert len(pairs) == 4
    assert ("ILLS-01", "MSTS-01") in pairs
    assert ("ILLS-01", "MSTS-03") in pairs
    assert ("ILLS-02", "MSTS-02") in pairs
    assert ("ILLS-02", "MSTS-03") in pairs
    assert ("ILLS-01", "MSTS-02") not in pairs


def test_is_valid_pair(simple_visibility_data):
    """Test checking if specific pairs are valid."""
    visibility = IlluminatorTelescopeVisibility(simple_visibility_data)

    assert visibility.is_valid_pair("ILLS-01", "MSTS-01") is True
    assert visibility.is_valid_pair("ILLS-01", "MSTS-02") is False
    assert visibility.is_valid_pair("ILLS-02", "MSTS-02") is True


def test_is_valid_pair_invalid_illuminator(simple_visibility_data):
    """Test checking pair with non-existent illuminator."""
    visibility = IlluminatorTelescopeVisibility(simple_visibility_data)

    with pytest.raises(ValueError, match="Illuminator 'INVALID' not found"):
        visibility.is_valid_pair("INVALID", "MSTS-01")


def test_is_valid_pair_invalid_telescope(simple_visibility_data):
    """Test checking pair with non-existent telescope."""
    visibility = IlluminatorTelescopeVisibility(simple_visibility_data)

    with pytest.raises(ValueError, match="Telescope 'INVALID' not found"):
        visibility.is_valid_pair("ILLS-01", "INVALID")


def test_get_telescopes_for_illuminator(simple_visibility_data):
    """Test getting telescopes for a specific illuminator."""
    visibility = IlluminatorTelescopeVisibility(simple_visibility_data)

    tels_01 = visibility.get_telescopes_for_illuminator("ILLS-01")
    assert len(tels_01) == 2
    assert "MSTS-01" in tels_01
    assert "MSTS-03" in tels_01

    tels_02 = visibility.get_telescopes_for_illuminator("ILLS-02")
    assert len(tels_02) == 2
    assert "MSTS-02" in tels_02
    assert "MSTS-03" in tels_02


def test_get_telescopes_for_invalid_illuminator(simple_visibility_data):
    """Test getting telescopes for non-existent illuminator."""
    visibility = IlluminatorTelescopeVisibility(simple_visibility_data)

    with pytest.raises(ValueError, match="Illuminator 'INVALID' not found"):
        visibility.get_telescopes_for_illuminator("INVALID")


def test_get_illuminators_for_telescope(simple_visibility_data):
    """Test getting illuminators for a specific telescope."""
    visibility = IlluminatorTelescopeVisibility(simple_visibility_data)

    ills_01 = visibility.get_illuminators_for_telescope("MSTS-01")
    assert len(ills_01) == 1
    assert "ILLS-01" in ills_01

    ills_03 = visibility.get_illuminators_for_telescope("MSTS-03")
    assert len(ills_03) == 2
    assert "ILLS-01" in ills_03
    assert "ILLS-02" in ills_03


def test_get_illuminators_for_invalid_telescope(simple_visibility_data):
    """Test getting illuminators for non-existent telescope."""
    visibility = IlluminatorTelescopeVisibility(simple_visibility_data)

    with pytest.raises(ValueError, match="Telescope 'INVALID' not found"):
        visibility.get_illuminators_for_telescope("INVALID")


def test_properties(simple_visibility_data):
    """Test property accessors."""
    visibility = IlluminatorTelescopeVisibility(simple_visibility_data)

    assert visibility.n_illuminators == 2
    assert visibility.n_telescopes == 3
    assert visibility.n_valid_pairs == 4


def test_north_site_structure(north_visibility_data):
    """Test with realistic North site structure."""
    visibility = IlluminatorTelescopeVisibility(north_visibility_data)

    assert visibility.n_illuminators == 2
    assert visibility.n_telescopes == 4

    # ILLN-02 should illuminate all telescopes
    tels = visibility.get_telescopes_for_illuminator("ILLN-02")
    assert len(tels) == 4

    # ILLN-01 should only illuminate MST telescopes
    tels = visibility.get_telescopes_for_illuminator("ILLN-01")
    assert len(tels) == 2
    assert "MSTN-01" in tels
    assert "MSTN-02" in tels

    # Check total valid pairs
    pairs = visibility.get_valid_pairs()
    assert len(pairs) == 6


def test_logging(simple_visibility_data, caplog):
    """Test that logging messages are generated correctly."""
    with caplog.at_level(logging.INFO):
        IlluminatorTelescopeVisibility(simple_visibility_data)

    assert "Reading illuminator visibility table" in caplog.text


def test_logging_valid_pairs(simple_visibility_data, caplog):
    """Test that __init__ logs valid pair count."""
    with caplog.at_level(logging.INFO):
        IlluminatorTelescopeVisibility(simple_visibility_data)

    assert "Found 4 valid illuminator-telescope pairs" in caplog.text
