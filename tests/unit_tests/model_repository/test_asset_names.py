"""Tests for simulation-model asset naming helpers."""

import pytest

from simtools.model_repository.asset_names import (
    SOURCE_VALUE_KEY,
    get_export_file_name,
    get_simtel_table_file_name,
    qualify_parameter_file_name,
)


@pytest.mark.parametrize(
    ("parameter_data", "fallback_instrument", "expected"),
    [
        ({"value": "table.ecsv", "instrument": "North-LST"}, None, "table-North-LST.ecsv"),
        ({"value": "table.ecsv"}, "South-MST", "table-South-MST.ecsv"),
        ({"value": "table.ecsv"}, None, "table-global.ecsv"),
        ({"value": "table.txt", "instrument": "North-LST"}, None, "table.txt"),
        ({"value": 42}, None, 42),
    ],
)
def test_get_export_file_name(parameter_data, fallback_instrument, expected):
    """Return qualified names only for ECSV values."""
    assert get_export_file_name(parameter_data, fallback_instrument) == expected


def test_get_export_file_name_uses_original_source_value():
    """Use the unqualified source value when metadata has already been exported."""
    parameter_data = {
        "value": "table-North-LST.ecsv",
        SOURCE_VALUE_KEY: "table.ecsv",
        "instrument": "North-LST",
    }

    assert get_export_file_name(parameter_data) == "table-North-LST.ecsv"


def test_get_export_file_name_can_preserve_source_name():
    """Allow simulator inputs that require the model-declared basename."""
    parameter_data = {
        "value": "atmospheric_profile-1.0.0.ecsv",
        "instrument": "OBS-South",
        "qualify_filename": False,
    }

    assert get_export_file_name(parameter_data) == "atmospheric_profile-1.0.0.ecsv"


def test_qualify_parameter_file_name_updates_ecsv_metadata():
    """Store the source value and update the exported value for an ECSV asset."""
    parameter_data = {"value": "table.ecsv", "instrument": "North-LST"}

    result = qualify_parameter_file_name(parameter_data)

    assert result == "table-North-LST.ecsv"
    assert parameter_data == {
        "value": "table-North-LST.ecsv",
        "instrument": "North-LST",
        SOURCE_VALUE_KEY: "table.ecsv",
    }


def test_get_simtel_table_file_name_is_shared_by_model_identity():
    """Native sim_telarray names do not depend on the telescope instance."""
    parameter = {
        "value": "secondary_mirror_reflectivity-1.0.0.ecsv",
        "instrument": "SSTS-design",
    }

    assert get_simtel_table_file_name(parameter) == (
        "secondary_mirror_reflectivity-1.0.0-SSTS-design.dat"
    )


def test_get_simtel_table_file_name_requires_identity_metadata():
    """Unscoped ECSV parameters retain the legacy per-config fallback."""
    assert get_simtel_table_file_name({"value": "table.ecsv"}) is None


@pytest.mark.parametrize(
    "parameter_data",
    [
        {
            "value": "table-North-LST.ecsv",
            SOURCE_VALUE_KEY: "table.ecsv",
            "instrument": "North-LST",
        },
        {"value": "table.txt"},
        {"value": 42},
        {"value": None},
    ],
)
def test_qualify_parameter_file_name_leaves_non_qualifiable_values_unchanged(parameter_data):
    """Leave already-exported, non-ECSV, and non-string values unchanged."""
    original = parameter_data.copy()
    value = parameter_data["value"]

    assert qualify_parameter_file_name(parameter_data) == value
    assert parameter_data == original
