"""Tests for contract-driven sim_telarray table serializers."""

import astropy.units as u
import pytest
from astropy.table import QTable

from simtools.simtel.table_serializers import (
    SimtelTableWriter,
    validate_simtel_serialization,
    write_simtel_table,
)


def _contract(table_format, columns, **kwargs):
    """Build a complete serializer contract for a test table."""
    return {
        "table_format": table_format,
        "columns": columns,
        "row_sort_keys": kwargs.pop("row_sort_keys", []),
        "float_format": kwargs.pop("float_format", ".1f"),
        "write_comments": False,
        "units": kwargs.pop("units", dict.fromkeys(columns, "dimensionless")),
        **kwargs,
    }


def test_write_simtel_table_sorts_plain_rows(tmp_test_directory):
    table = QTable({"time": [2.0, 1.0], "amplitude": [0.2, 0.1]})
    contract = _contract("plain", ["time", "amplitude"], row_sort_keys=["time"])

    result = write_simtel_table(table, tmp_test_directory, contract, "pulse.dat")

    assert result == "pulse.dat"
    assert (tmp_test_directory / result).read_text(encoding="utf-8").splitlines() == [
        "1.0 0.1",
        "2.0 0.2",
    ]


def test_write_simtel_table_rejects_missing_contract():
    with pytest.raises(ValueError, match="requires a serialization contract"):
        write_simtel_table(QTable({"value": [1]}), ".", None)


def test_write_simtel_table_uses_registered_writer(tmp_test_directory):
    table = QTable({"value": [1.0]})
    contract = _contract("plain", ["value"])

    result = SimtelTableWriter.write(table, tmp_test_directory, contract, "value.dat")

    assert result == "value.dat"
    assert (tmp_test_directory / result).read_text(encoding="utf-8") == "1.0\n"


def test_validate_simtel_serialization_checks_units():
    table = QTable({"wavelength": [400.0] * u.nm})
    contract = _contract("plain", ["wavelength"], units={"wavelength": "deg"})

    with pytest.raises(ValueError, match="expected deg"):
        validate_simtel_serialization(table, contract)


def test_write_simtel_table_serializes_atmospheric_matrix(tmp_test_directory):
    table = QTable(
        {
            "wavelength": [300.0, 300.0] * u.nm,
            "altitude": [1.0, 2.0] * u.km,
            "extinction": [0.1, 0.2],
        }
    )
    table.meta["observatory_level"] = 1.5 * u.km
    contract = _contract(
        "atmospheric_transmission",
        ["wavelength", "altitude", "extinction"],
        row_sort_keys=["wavelength", "altitude"],
        matrix_axes=["wavelength", "altitude"],
        value_column="extinction",
        units={"wavelength": "nm", "altitude": "km", "extinction": "dimensionless"},
    )

    write_simtel_table(table, tmp_test_directory, contract, "atmosphere.dat")

    assert (tmp_test_directory / "atmosphere.dat").read_text(encoding="utf-8").splitlines() == [
        "# H2= 1.5, H1= 1.0 2.0",
        "300.0 0.1 0.2",
    ]
