#!/usr/bin/python3

"""Unit tests for illuminator_visibility module."""

import pytest

from simtools.model.illuminator_visibility import IlluminatorTelescopeVisibility


@pytest.fixture
def simple_visibility_data():
    """Create structured visibility records."""
    return [
        {"illuminator_id": "ILLS-01", "telescope_id": "MSTS-01", "visible": True},
        {"illuminator_id": "ILLS-01", "telescope_id": "MSTS-02", "visible": False},
        {"illuminator_id": "ILLS-01", "telescope_id": "MSTS-03", "visible": True},
        {"illuminator_id": "ILLS-02", "telescope_id": "MSTS-01", "visible": False},
        {"illuminator_id": "ILLS-02", "telescope_id": "MSTS-02", "visible": True},
        {"illuminator_id": "ILLS-02", "telescope_id": "MSTS-03", "visible": True},
    ]


@pytest.fixture
def north_visibility_data():
    """Create structured records matching North site structure."""
    return [
        {"illuminator_id": "ILLN-01", "telescope_id": "LSTN-01", "visible": False},
        {"illuminator_id": "ILLN-01", "telescope_id": "LSTN-02", "visible": False},
        {"illuminator_id": "ILLN-01", "telescope_id": "MSTN-01", "visible": True},
        {"illuminator_id": "ILLN-01", "telescope_id": "MSTN-02", "visible": True},
        {"illuminator_id": "ILLN-02", "telescope_id": "LSTN-01", "visible": True},
        {"illuminator_id": "ILLN-02", "telescope_id": "LSTN-02", "visible": True},
        {"illuminator_id": "ILLN-02", "telescope_id": "MSTN-01", "visible": True},
        {"illuminator_id": "ILLN-02", "telescope_id": "MSTN-02", "visible": True},
    ]


def test_init_valid_data(simple_visibility_data):
    visibility = IlluminatorTelescopeVisibility(simple_visibility_data)
    assert visibility is not None
    assert visibility.n_illuminators == 2
    assert visibility.n_telescopes == 3


def test_init_invalid_type():
    with pytest.raises(ValueError, match="Expected a list"):
        IlluminatorTelescopeVisibility("not a dict")


def test_init_missing_keys():
    with pytest.raises(ValueError, match="must contain illuminator_id"):
        IlluminatorTelescopeVisibility([{"columns": ["a", "b", "c"]}])


def test_init_missing_required_columns():
    data = [{"wrong_column": "ILL-01", "telescope_id": "TEL-01", "visible": True}]
    with pytest.raises(ValueError, match="must contain illuminator_id"):
        IlluminatorTelescopeVisibility(data)


def test_init_empty_rows():
    data = []
    with pytest.raises(ValueError, match="contains no illuminators"):
        IlluminatorTelescopeVisibility(data)


def test_get_illuminators(simple_visibility_data):
    visibility = IlluminatorTelescopeVisibility(simple_visibility_data)
    illuminators = visibility.get_illuminators()

    assert len(illuminators) == 2
    assert "ILLS-01" in illuminators
    assert "ILLS-02" in illuminators


def test_get_telescopes(simple_visibility_data):
    visibility = IlluminatorTelescopeVisibility(simple_visibility_data)
    telescopes = visibility.get_telescopes()

    assert len(telescopes) == 3
    assert "MSTS-01" in telescopes
    assert "MSTS-02" in telescopes
    assert "MSTS-03" in telescopes


def test_get_valid_pairs(simple_visibility_data):
    visibility = IlluminatorTelescopeVisibility(simple_visibility_data)
    pairs = visibility.get_valid_pairs()

    assert len(pairs) == 4
    assert ("ILLS-01", "MSTS-01") in pairs
    assert ("ILLS-01", "MSTS-03") in pairs
    assert ("ILLS-02", "MSTS-02") in pairs
    assert ("ILLS-02", "MSTS-03") in pairs
    assert ("ILLS-01", "MSTS-02") not in pairs


def test_is_valid_pair(simple_visibility_data):
    visibility = IlluminatorTelescopeVisibility(simple_visibility_data)

    assert visibility.is_valid_pair("ILLS-01", "MSTS-01") is True
    assert visibility.is_valid_pair("ILLS-01", "MSTS-02") is False
    assert visibility.is_valid_pair("ILLS-02", "MSTS-02") is True


def test_is_valid_pair_invalid_illuminator(simple_visibility_data):
    visibility = IlluminatorTelescopeVisibility(simple_visibility_data)

    with pytest.raises(ValueError, match="Illuminator 'INVALID' not found"):
        visibility.is_valid_pair("INVALID", "MSTS-01")


def test_is_valid_pair_invalid_telescope(simple_visibility_data):
    visibility = IlluminatorTelescopeVisibility(simple_visibility_data)

    with pytest.raises(ValueError, match="Telescope 'INVALID' not found"):
        visibility.is_valid_pair("ILLS-01", "INVALID")


def test_get_telescopes_for_illuminator(simple_visibility_data):
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
    visibility = IlluminatorTelescopeVisibility(simple_visibility_data)

    with pytest.raises(ValueError, match="Illuminator 'INVALID' not found"):
        visibility.get_telescopes_for_illuminator("INVALID")


def test_get_illuminators_for_telescope(simple_visibility_data):
    visibility = IlluminatorTelescopeVisibility(simple_visibility_data)

    ills_01 = visibility.get_illuminators_for_telescope("MSTS-01")
    assert len(ills_01) == 1
    assert "ILLS-01" in ills_01

    ills_03 = visibility.get_illuminators_for_telescope("MSTS-03")
    assert len(ills_03) == 2
    assert "ILLS-01" in ills_03
    assert "ILLS-02" in ills_03


def test_get_illuminators_for_invalid_telescope(simple_visibility_data):
    visibility = IlluminatorTelescopeVisibility(simple_visibility_data)

    with pytest.raises(ValueError, match="Telescope 'INVALID' not found"):
        visibility.get_illuminators_for_telescope("INVALID")


def test_properties(simple_visibility_data):
    visibility = IlluminatorTelescopeVisibility(simple_visibility_data)

    assert visibility.n_illuminators == 2
    assert visibility.n_telescopes == 3
    assert visibility.n_valid_pairs == 4
