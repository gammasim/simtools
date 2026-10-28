"""Tests for schema-driven model parameter builders."""

from pathlib import Path

import pytest
from astropy.table import Table

from simtools.data_model import schema
from simtools.data_model.validate_data import DataValidator
from tests.unit_tests.builders.parameters import parameter

_SCHEMA_DIRECTORY = Path(__file__).parents[3] / "src" / "simtools" / "schemas" / "model_parameters"


@pytest.mark.parametrize(
    "name",
    [
        path.name.removesuffix(".schema.yml")
        for path in sorted(_SCHEMA_DIRECTORY.glob("*.schema.yml"))
    ],
)
def test_every_schema_is_buildable(name):
    """Build a value for every packaged model parameter schema."""
    record = parameter(name)
    description = schema.get_model_parameter_schema(name)
    if description["data"][0]["type"] == "data_table":
        assert isinstance(record, Table)
        return

    validated = DataValidator.validate_model_parameter(record)

    assert validated["parameter"] == name
    assert validated["model_parameter_schema_version"] == description["schema_version"]


def test_parameter_uses_latest_schema_version_and_its_default():
    """The newest schema document is used, here with its extended allowed range."""
    record = parameter("store_photoelectrons")

    assert record["model_parameter_schema_version"] == "0.2.0"
    assert record["value"] == -1


def test_parameter_sets_one_unit_per_value():
    """Multi-value parameters carry the unit declared for each value."""
    description = schema.get_model_parameter_schema("focus_offset")
    record = parameter("focus_offset")

    assert record["unit"] == [datum.get("unit") for datum in description["data"]]
    assert len(record["value"]) == len(description["data"])


def test_parameter_builds_single_item_json_arrays_without_extra_nesting():
    """A JSON-schema array parameter is a flat list of objects."""
    record = parameter("camera_pixel_types")

    assert isinstance(record["value"], list)
    assert isinstance(record["value"][0], dict)
    DataValidator.validate_model_parameter(dict(record))


def test_parameter_uses_valid_values_for_custom_string_formats():
    """Strings with a custom format get a value the format checker accepts."""
    layout = parameter("array_layouts")["value"][0]

    assert layout["elements"] == ["LSTN-01"]


def test_parameter_with_explicit_value_site_and_instrument():
    """Explicit values and identity fields are kept and validated."""
    record = parameter(
        "camera_body_diameter", value=1.5, site="North", instrument="LSTN-01", version="2.0.0"
    )

    assert record["value"] == pytest.approx(1.5)
    assert record["site"] == "North"
    assert record["instrument"] == "LSTN-01"
    assert record["parameter_version"] == "2.0.0"


def test_parameter_rejects_values_outside_the_schema():
    """An explicit value is validated against the schema."""
    with pytest.raises(ValueError, match="out of range"):
        parameter("store_photoelectrons", value=-2)


def test_file_parameter_value_is_a_file_name():
    """File-valued parameters reference a file named after the parameter."""
    record = parameter("mirror_list")

    assert record["file"] is True
    assert record["value"].startswith("mirror_list")
