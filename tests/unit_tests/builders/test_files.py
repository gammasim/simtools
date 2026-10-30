"""Round-trip tests for generated model-parameter files."""

from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.tests.helper import assert_quantity_allclose

from simtools.constants import SCHEMA_PATH
from simtools.data_model import schema
from simtools.data_model.table_asset import read_ecsv_asset
from simtools.model.camera import Camera
from simtools.model.mirrors import Mirrors
from tests.unit_tests.builders.files import camera_configuration_file, table_file

_TABLE_PARAMETERS = [
    path.name.removesuffix(".schema.yml")
    for path in sorted((SCHEMA_PATH / "model_parameters").glob("*.schema.yml"))
    if (
        entry := schema.get_model_parameter_schema(path.name.removesuffix(".schema.yml"))["data"][0]
    )["type"]
    == "file"
    and entry.get("table_columns")
]


@pytest.mark.parametrize("name", _TABLE_PARAMETERS)
def test_every_table_file_round_trips(name, tmp_test_directory):
    """Every current file table has one schema-valid row with units and descriptions."""
    destination = Path(tmp_test_directory) / f"{name}.ecsv"
    entry = schema.get_model_parameter_schema(name)["data"][0]

    path = table_file(name, destination)
    table = read_ecsv_asset(path, schema_entry=entry)

    assert path == destination
    assert len(table) == 1
    assert table.colnames == [column["name"] for column in entry["table_columns"]]
    for column in entry["table_columns"]:
        actual = table[column["name"]]
        assert actual.info.description == column["description"]
        if column.get("unit") == "dimensionless":
            assert actual.unit in (None, u.dimensionless_unscaled)
        elif column.get("unit"):
            assert actual.unit == u.Unit(column["unit"])
        if column["type"] != "string":
            assert actual.dtype == np.dtype(column["type"])
            assert np.all(np.isfinite(actual))
        if "default" in column:
            np.testing.assert_array_equal(np.asarray(actual), [column["default"]])
        elif column.get("allowed_values"):
            assert actual[0] in column["allowed_values"]


@pytest.mark.parametrize(
    "rows",
    [
        [(300.0, 0.0, 0.25), (600.0, 20.0, 0.75)],
        [
            {"wavelength": 300.0, "incidence_angle": 0.0, "transmission": 0.25},
            {"wavelength": 600.0, "incidence_angle": 20.0, "transmission": 0.75},
        ],
    ],
)
def test_table_file_preserves_explicit_rows(rows, tmp_test_directory):
    """Row tuples and dictionaries preserve physical values and schema units."""
    entry = schema.get_model_parameter_schema("camera_filter")["data"][0]
    path = table_file("camera_filter", Path(tmp_test_directory) / "filter.ecsv", rows=rows)

    table = read_ecsv_asset(path, schema_entry=entry)

    assert_quantity_allclose(table["wavelength"], [300.0, 600.0] * u.nm)
    assert_quantity_allclose(table["incidence_angle"], [0.0, 20.0] * u.deg)
    assert_quantity_allclose(table["transmission"], [0.25, 0.75] * u.dimensionless_unscaled)


def test_mirror_table_file_round_trips_through_mirrors(tmp_test_directory):
    """A mirror asset can be used directly by the mirror geometry parser."""
    path = table_file(
        "mirror_list",
        Path(tmp_test_directory) / "mirrors.ecsv",
        rows=[(10.0, -20.0, 120.0, 1600.0, 1.0, 0.0, 7)],
    )

    mirrors = Mirrors(path)

    assert mirrors.number_of_mirrors == 1
    for actual, expected in zip(
        mirrors.get_single_mirror_parameters(7),
        (10 * u.cm, -20 * u.cm, 120 * u.cm, 1600 * u.cm, 1 * u.dimensionless_unscaled),
        strict=True,
    ):
        assert_quantity_allclose(actual, expected)
    assert mirrors.mirror_table["mirror_panel_id"].tolist() == [7]


@pytest.mark.parametrize("name", ["asum_shaping", "mirror_class", "array_coordinates_UTM"])
def test_table_file_requires_a_file_schema_with_columns(name, tmp_test_directory):
    """Parameters without file-table definitions cannot be written as table assets."""
    with pytest.raises(ValueError, match="file parameter with table_columns"):
        table_file(name, Path(tmp_test_directory) / "invalid.ecsv")


def test_table_file_rejects_out_of_range_rows(tmp_test_directory):
    """An explicit row must satisfy the parameter schema."""
    with pytest.raises(ValueError, match=r"transmission.*allowed_range\.max"):
        table_file(
            "camera_filter",
            Path(tmp_test_directory) / "invalid.ecsv",
            rows=[(400.0, 0.0, 1.5)],
        )


def test_table_file_requires_ecsv_destination(tmp_test_directory):
    """Generated model-table files use the extension required by the asset parser."""
    with pytest.raises(ValueError, match=r"ECSV.*\.ecsv"):
        table_file("mirror_list", Path(tmp_test_directory) / "mirrors.dat")


def test_table_file_rejects_empty_rows(tmp_test_directory):
    """A minimal model-table file contains at least one row."""
    with pytest.raises(ValueError, match="at least one row"):
        table_file("mirror_list", Path(tmp_test_directory) / "empty.ecsv", rows=[])


def test_camera_configuration_file_round_trips(tmp_test_directory):
    """The native camera writer provides a complete pixel type and one enabled pixel."""
    destination = Path(tmp_test_directory) / "camera.dat"

    path = camera_configuration_file(destination)
    pixels = Camera.read_pixel_list(path)

    assert path == destination
    assert pixels["pix_id"] == [0]
    assert pixels["pix_on"] == [True]
    np.testing.assert_allclose(pixels["x"], [0.0], atol=1e-12)
    np.testing.assert_allclose(pixels["y"], [0.0], atol=1e-12)
    assert pixels["pixel_shape"] == 1
    assert pixels["pixel_diameter"] == pytest.approx(1.0)
    assert pixels["rotate_angle"] == pytest.approx(0.0, abs=1e-12)
    assert pixels["lightguide_efficiency_angle_file"] == "none"
    assert pixels["lightguide_efficiency_wavelength_file"] == "none"
