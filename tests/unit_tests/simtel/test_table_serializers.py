"""Tests for contract-driven sim_telarray table serializers."""

import astropy.units as u
import pytest
from astropy.table import QTable

from simtools.simtel.table_serializers import (
    SimtelTableWriter,
    _unit_matches,
    _validate_contract_definition,
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


def test_write_simtel_table_rejects_non_table_and_unsafe_name(tmp_test_directory):
    contract = _contract("plain", ["value"])
    with pytest.raises(TypeError, match="requires an Astropy table"):
        SimtelTableWriter.write({"value": [1.0]}, tmp_test_directory, contract, "value.dat")

    with pytest.raises(ValueError, match="Unsafe"):
        SimtelTableWriter.write(
            QTable({"value": [1.0]}), tmp_test_directory, contract, "../value.dat"
        )


def test_write_simtel_table_rejects_unknown_format(tmp_test_directory):
    with pytest.raises(ValueError, match="Unknown sim_telarray table format"):
        SimtelTableWriter.write(
            QTable({"value": [1.0]}),
            tmp_test_directory,
            _contract("unknown", ["value"]),
            "value.dat",
        )


def test_write_simtel_table_uses_metadata_default_name(tmp_test_directory):
    table = QTable({"value": [1.0]})
    table.meta["parameter_name"] = "metadata_name"

    assert write_simtel_table(table, tmp_test_directory, _contract("plain", ["value"])) == (
        "metadata_name.dat"
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda contract: contract.update(columns=[]), "unique and non-empty"),
        (lambda contract: contract.update(columns=["value", "value"]), "unique and non-empty"),
        (lambda contract: contract.update(allowed_columns=[]), "allowed columns"),
        (lambda contract: contract.update(row_sort_keys=["missing"]), "sort keys"),
        (lambda contract: contract.update(matrix_axes=["value"]), "matrix_axes"),
        (lambda contract: contract.update(matrix_axes=["value", "missing"]), "matrix_axes"),
        (
            lambda contract: contract.update(
                matrix_axes=["value", "extra"],
                allowed_columns=["value", "extra"],
                value_column="missing",
            ),
            "value_column",
        ),
    ],
)
def test_contract_definition_rejects_invalid_references(change, message):
    contract = _contract("plain", ["value"], row_sort_keys=["value"])
    change(contract)

    with pytest.raises(ValueError, match=message):
        _validate_contract_definition(contract)


def test_validate_simtel_serialization_rejects_missing_and_unexpected_columns():
    table = QTable({"value": [1.0], "extra": [2.0]})
    with pytest.raises(ValueError, match="undeclared columns"):
        validate_simtel_serialization(table, _contract("plain", ["value"]))

    with pytest.raises(ValueError, match="missing columns"):
        validate_simtel_serialization(QTable({"extra": [2.0]}), _contract("plain", ["value"]))


def test_validate_simtel_serialization_handles_optional_and_rpol_columns():
    table = QTable({"wavelength": [400.0] * u.nm, "response": [0.5]})
    optional_contract = _contract(
        "plain",
        ["wavelength"],
        optional_columns=["response"],
        units={"wavelength": "nm", "response": "dimensionless"},
    )
    validate_simtel_serialization(table, optional_contract)

    rpol_contract = _contract(
        "rpol_matrix",
        ["wavelength", "response"],
        matrix_axes=["wavelength", "angle"],
        value_column="response",
        units={"wavelength": "nm", "response": "dimensionless"},
    )
    validate_simtel_serialization(table, rpol_contract)


def test_validate_simtel_serialization_rejects_incomplete_matrix():
    table = QTable(
        {
            "wavelength": [400.0, 500.0] * u.nm,
            "angle": [0.0, 10.0] * u.deg,
            "response": [0.5, 0.4],
        }
    )
    contract = _contract(
        "plain",
        ["wavelength", "angle", "response"],
        matrix_axes=["wavelength", "angle"],
        value_column="response",
        units={"wavelength": "nm", "angle": "deg", "response": "dimensionless"},
    )
    with pytest.raises(ValueError, match="complete Cartesian grid"):
        validate_simtel_serialization(table, contract)

    contract["missing_value"] = -1.0
    validate_simtel_serialization(table, contract)


def test_validate_simtel_serialization_rejects_duplicate_matrix_rows():
    table = QTable(
        {
            "wavelength": [400.0, 400.0] * u.nm,
            "angle": [0.0, 0.0] * u.deg,
            "response": [0.5, 0.5],
        }
    )
    contract = _contract(
        "plain",
        ["wavelength", "angle", "response"],
        matrix_axes=["wavelength", "angle"],
        value_column="response",
        units={"wavelength": "nm", "angle": "deg", "response": "dimensionless"},
    )
    with pytest.raises(ValueError, match="duplicate axis"):
        validate_simtel_serialization(table, contract)


@pytest.mark.parametrize(
    ("actual", "expected", "matches"),
    [
        (None, "dimensionless", True),
        (u.dimensionless_unscaled, "dimensionless", True),
        (u.nm, "dimensionless", False),
        (u.nm, "nm", True),
    ],
)
def test_unit_matches_dimensionless_and_equivalent_units(actual, expected, matches):
    assert _unit_matches(actual, expected) is matches


def test_unit_matches_rejects_invalid_units():
    with pytest.raises(ValueError, match="Invalid table unit"):
        _unit_matches("not-a-unit", "nm")

    with pytest.raises(ValueError, match="Invalid table unit"):
        _unit_matches(u.nm, "not-a-unit")


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


def test_write_simtel_table_serializes_atmosphere_without_observatory_level(tmp_test_directory):
    table = QTable(
        {
            "wavelength": [300.0] * u.nm,
            "altitude": [1.0] * u.km,
            "extinction": [0.1],
        }
    )
    contract = _contract(
        "atmospheric_transmission",
        ["wavelength", "altitude", "extinction"],
        matrix_axes=["wavelength", "altitude"],
        value_column="extinction",
        units={"wavelength": "nm", "altitude": "km", "extinction": "dimensionless"},
    )

    write_simtel_table(table, tmp_test_directory, contract, "atmosphere.dat")

    assert (tmp_test_directory / "atmosphere.dat").read_text(encoding="utf-8") == (
        "# H1= 1.0\n300.0 0.1\n"
    )


def test_write_simtel_table_serializes_one_dimensional_rpol(tmp_test_directory):
    table = QTable({"wavelength": [500.0, 400.0] * u.nm, "response": [0.5, 0.4]})
    contract = _contract(
        "rpol_matrix",
        ["wavelength", "response"],
        matrix_axes=["wavelength", "angle"],
        value_column="response",
        row_sort_keys=["wavelength"],
        allowed_columns=["wavelength", "angle", "response"],
        units={"wavelength": "nm", "response": "dimensionless"},
    )

    write_simtel_table(table, tmp_test_directory, contract, "response.dat")

    assert (tmp_test_directory / "response.dat").read_text(encoding="utf-8").splitlines() == [
        "400.0 0.4",
        "500.0 0.5",
    ]


def test_write_simtel_table_serializes_two_dimensional_rpol(tmp_test_directory):
    table = QTable(
        {
            "wavelength": [400.0, 400.0, 500.0, 500.0] * u.nm,
            "angle": [0.0, 10.0, 0.0, 10.0] * u.deg,
            "response": [0.5, 0.4, 0.3, 0.2],
        }
    )
    contract = _contract(
        "rpol_matrix",
        ["wavelength", "angle", "response"],
        matrix_axes=["wavelength", "angle"],
        value_column="response",
        units={"wavelength": "nm", "angle": "deg", "response": "dimensionless"},
    )

    write_simtel_table(table, tmp_test_directory, contract, "response.dat")

    assert (tmp_test_directory / "response.dat").read_text(encoding="utf-8").splitlines() == [
        "#@RPOL@[ANGLE=] 2",
        "ANGLE= 0.0 10.0",
        "400.0 0.5 0.4",
        "500.0 0.3 0.2",
    ]
