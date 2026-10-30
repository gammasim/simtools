"""Tests for model parameter catalog helpers."""

import pytest

from simtools.data_model import parameter_catalog as catalog


def test_parameter_collection_uses_schema_metadata():
    assert catalog.get_collection_name_from_parameter_name("atmospheric_profile") == "sites"
    with pytest.raises(KeyError, match="Parameter Not_a_parameter without schema definition"):
        catalog.get_collection_name_from_parameter_name("Not_a_parameter")


def test_scope_helpers():
    assert catalog.get_model_parameter_scope("configuration_corsika", "MSTN-01", "x") == "global"
    assert catalog.get_model_parameter_scope("telescopes", "MSTN-01", "x") == "MSTN-01"
    assert catalog.get_sim_telarray_parameter_scopes("MSTN-01", "MSTN-FlashCam") == [
        "global",
        "MSTN-FlashCam",
        "MSTN-01",
    ]
    with pytest.raises(KeyError, match="Missing design model for MSTN-01"):
        catalog.get_sim_telarray_parameter_scopes("MSTN-01")


def test_simtel_scope_fallbacks_and_software_catalog():
    assert catalog.get_sim_telarray_parameter_scopes(None) == ["global"]
    assert catalog.get_sim_telarray_parameter_scopes("MSTN-FlashCam") == [
        "global",
        "MSTN-FlashCam",
    ]
    assert catalog.get_sim_telarray_parameter_scopes(
        "MSTN-01", ignore_missing_design_model=True
    ) == ["global", "MSTN-01", "MSTN-01", "MSTN-design"]
    assert "sim_telarray" in catalog.simulation_software()


def test_metadata_mode_for_missing_and_unmapped_parameter_software():
    with pytest.raises(KeyError, match="Parameter Not_a_parameter without schema definition"):
        catalog.get_simulation_software_meta_parameter_mode("Not_a_parameter")
    assert catalog.get_simulation_software_meta_parameter_mode("focal_length", "unknown") is None
