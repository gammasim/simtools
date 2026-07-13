"""Tests for shared histogram output metadata helpers."""

import astropy.units as u
import pytest
from astropy.table import Table

from simtools.production_configuration.histogram_output_metadata import (
    extract_histogram_output_metadata,
)


def test_extract_histogram_output_metadata_from_dict_preserves_available_values():
    source = {
        "primary_particle": "gamma",
        "zenith": 20.0 * u.deg,
        "azimuth": 180.0 * u.deg,
        "nsb_level": 1.0,
        "energy_min": 0.1 * u.TeV,
    }

    result = extract_histogram_output_metadata(
        source,
        {
            "primary_particle": "primary_particle",
            "zenith": "zenith",
            "azimuth": "azimuth",
            "nsb_level": "nsb_level",
            "br_energy_min": "energy_min",
            "br_viewcone_max": "viewcone_max",
        },
    )

    assert result == {
        "primary_particle": "gamma",
        "zenith": 20.0 * u.deg,
        "azimuth": 180.0 * u.deg,
        "nsb_level": 1.0,
        "br_energy_min": 0.1 * u.TeV,
    }


def test_extract_histogram_output_metadata_from_table_row_reconstructs_quantities():
    metadata = Table(
        rows=[
            {
                "array_name": "alpha",
                "primary_particle": "gamma",
                "zenith": 20.0 * u.deg,
                "azimuth": 180.0 * u.deg,
                "nsb_level": 1.0,
                "energy_min": 0.1 * u.TeV,
            }
        ]
    )

    result = extract_histogram_output_metadata(
        metadata[0],
        {
            "primary_particle": "primary_particle",
            "zenith": "zenith",
            "azimuth": "azimuth",
            "nsb_level": "nsb_level",
            "br_energy_min": "energy_min",
        },
        include_array_name=True,
    )

    assert result["array_name"] == "alpha"
    assert result["primary_particle"] == "gamma"
    assert result["zenith"].to_value(u.deg) == pytest.approx(20.0)
    assert result["azimuth"].to_value(u.deg) == pytest.approx(180.0)
    assert result["nsb_level"] == pytest.approx(1.0)
    assert result["br_energy_min"].to_value(u.TeV) == pytest.approx(0.1)
