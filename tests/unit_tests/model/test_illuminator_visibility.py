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


def test_init_normalizes_row_table_data():
    visibility = IlluminatorTelescopeVisibility(
        {
            "columns": ["illuminator_id", "telescope_id", "visible"],
            "rows": [["ILLN-01", "MSTN-02", True]],
        }
    )

    assert visibility.get_valid_pairs() == [("ILLN-01", "MSTN-02")]


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


def test_properties(simple_visibility_data):
    visibility = IlluminatorTelescopeVisibility(simple_visibility_data)

    assert visibility.n_telescopes == 3
    assert visibility.n_valid_pairs == 4
