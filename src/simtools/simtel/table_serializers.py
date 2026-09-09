"""Contract-driven serializers for sim_telarray tabular inputs."""

from itertools import product
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import Table


class SimtelTableWriter:
    """Write validated model tables using a declared sim_telarray contract."""

    _writers = {}

    @classmethod
    def register(cls, format_name):
        """Register a table-format adapter."""

        def decorator(writer):
            cls._writers[format_name] = writer
            return writer

        return decorator

    @classmethod
    def write(cls, table, destination, contract, output_name):
        """Validate and serialize one table using its schema contract."""
        if not isinstance(table, Table):
            raise TypeError(
                f"sim_telarray table writer requires an Astropy table, got {type(table)}"
            )
        _validate_contract_definition(contract)
        validate_simtel_serialization(table, contract)
        output_path = Path(destination) / output_name
        if output_path.name != output_name:
            raise ValueError(f"Unsafe sim_telarray output filename: {output_name}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            writer = cls._writers[contract["table_format"]]
        except KeyError as exc:
            raise ValueError(
                f"Unknown sim_telarray table format: {contract['table_format']}"
            ) from exc
        writer.write(table, output_path, contract)
        return output_path.name


def write_simtel_table(table, destination, contract, output_name=None):
    """Write a table with its required schema serialization contract."""
    if contract is None:
        raise ValueError("sim_telarray table writing requires a serialization contract")
    output_name = output_name or f"{table.meta.get('parameter_name', 'table')}.dat"
    return SimtelTableWriter.write(table, destination, dict(contract), output_name)


def validate_simtel_serialization(table, contract):
    """Validate positional, unit, and matrix invariants before serialization."""
    _validate_contract_columns(table, contract)
    _validate_contract_units(table, contract)
    _validate_contract_matrix(table, contract)


def _validate_contract_definition(contract):
    """Validate a complete schema-declared serialization contract."""
    required = {
        "table_format",
        "columns",
        "row_sort_keys",
        "float_format",
        "write_comments",
        "units",
    }
    missing = sorted(required - set(contract))
    if missing:
        raise ValueError(f"Incomplete sim_telarray serialization contract: {missing}")
    columns = contract["columns"]
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("sim_telarray serialization columns must be unique and non-empty")
    allowed = set(contract.get("allowed_columns", columns)) | set(
        contract.get("optional_columns", [])
    )
    if not set(columns) <= allowed:
        raise ValueError("sim_telarray serialization allowed columns must include columns")
    if not set(contract["row_sort_keys"]) <= allowed:
        raise ValueError("sim_telarray serialization sort keys must reference declared columns")
    axes = contract.get("matrix_axes", [])
    if len(axes) not in (0, 2) or len(axes) != len(set(axes)) or not set(axes) <= allowed:
        raise ValueError("sim_telarray serialization matrix_axes must contain two declared axes")
    if axes and contract.get("value_column") not in allowed:
        raise ValueError("sim_telarray serialization matrix requires a declared value_column")


def _validate_contract_columns(table, contract):
    """Validate table columns against its positional contract."""
    missing = sorted(set(contract["columns"]) - set(table.colnames))
    if missing:
        raise ValueError(f"sim_telarray serialization is missing columns: {missing}")
    allowed = set(contract.get("allowed_columns", contract["columns"]))
    allowed.update(contract.get("optional_columns", []))
    unexpected = sorted(set(table.colnames) - allowed)
    if unexpected:
        raise ValueError(f"sim_telarray serialization has undeclared columns: {unexpected}")
    if contract.get("matrix_axes"):
        required = set(contract["matrix_axes"]) | {contract["value_column"]}
        missing = sorted(required - set(table.colnames))
        one_dimensional_rpol = contract["table_format"] == "rpol_matrix" and missing == [
            contract["matrix_axes"][1]
        ]
        if missing and not one_dimensional_rpol:
            raise ValueError(f"sim_telarray serialization is missing columns: {missing}")


def _validate_contract_units(table, contract):
    """Validate canonical units required by the schema contract."""
    for name, expected in contract["units"].items():
        if name not in table.colnames:
            continue
        actual = getattr(table[name], "unit", None)
        if not _unit_matches(actual, expected):
            raise ValueError(f"sim_telarray column '{name}' has unit {actual}; expected {expected}")


def _unit_matches(actual, expected):
    """Return whether an actual column unit satisfies a contract unit."""
    if actual is None:
        return expected == "dimensionless"
    try:
        actual_unit = u.Unit(actual)
        if expected == "dimensionless":
            return actual_unit.is_equivalent(u.dimensionless_unscaled)
        return actual_unit == u.Unit(expected)
    except (TypeError, ValueError, u.UnitsError) as exc:
        raise ValueError(f"Invalid table unit {actual}; expected {expected}") from exc


def _validate_contract_matrix(table, contract):
    """Validate a complete unique Cartesian matrix when its axes are present."""
    axes = contract.get("matrix_axes", [])
    if not axes or axes[1] not in table.colnames:
        return
    keys = list(zip(*(_raw_values(table[axis]) for axis in axes), strict=True))
    if len(keys) != len(set(keys)):
        raise ValueError("sim_telarray matrix contains duplicate axis combinations")
    expected = set(product(*(sorted(set(_raw_values(table[axis]))) for axis in axes)))
    missing = expected - set(keys)
    if missing and "missing_value" not in contract:
        raise ValueError("sim_telarray matrix must contain a complete Cartesian grid")


def _ordered_table(table, contract):
    """Select schema-declared columns and canonical row order without mutation."""
    columns = list(contract["columns"])
    columns.extend(name for name in contract.get("optional_columns", []) if name in table.colnames)
    result = table[columns].copy(copy_data=True)
    sort_keys = [name for name in contract["row_sort_keys"] if name in result.colnames]
    sort_keys.extend(name for name in result.colnames if name not in sort_keys)
    if sort_keys:
        result.sort(sort_keys)
    return result


def _format(value, contract):
    """Format a scalar with the schema-declared numeric format."""
    value = getattr(value, "value", value)
    if isinstance(value, (float, np.floating)):
        return format(float(value), contract["float_format"])
    return str(value)


def _raw_values(values):
    """Return scalar values from an Astropy column."""
    return [getattr(value, "value", value) for value in values]


@SimtelTableWriter.register("plain")
@SimtelTableWriter.register("pulse")
@SimtelTableWriter.register("mirror_list")
class PlainTableWriter:
    """Serialize positional whitespace-separated tables."""

    @staticmethod
    def write(table, output_path, contract):
        """Write the selected sorted rows."""
        ordered = _ordered_table(table, contract)
        with output_path.open("w", encoding="utf-8") as file:
            for row in ordered:
                file.write(
                    " ".join(_format(row[name], contract) for name in ordered.colnames) + "\n"
                )


@SimtelTableWriter.register("rpol_matrix")
class RpolMatrixWriter:
    """Serialize wavelength and incidence-angle response matrices."""

    @staticmethod
    def write(table, output_path, contract):
        """Write either a one-dimensional response or a rectangular RPOL matrix."""
        axis_0, axis_1 = contract["matrix_axes"]
        if axis_1 not in table.colnames:
            PlainTableWriter.write(table, output_path, contract)
            return
        value_column = contract["value_column"]
        values = {(_scalar(row[axis_0]), _scalar(row[axis_1])): row[value_column] for row in table}
        axis_0_values = sorted(set(_raw_values(table[axis_0])))
        axis_1_values = sorted(set(_raw_values(table[axis_1])))
        with output_path.open("w", encoding="utf-8") as file:
            file.write("#@RPOL@[ANGLE=] 2\n")
            file.write(
                "ANGLE= " + " ".join(_format(value, contract) for value in axis_1_values) + "\n"
            )
            for axis_0_value in axis_0_values:
                fields = [_format(axis_0_value, contract)]
                fields.extend(
                    _format(values[(axis_0_value, axis_1_value)], contract)
                    for axis_1_value in axis_1_values
                )
                file.write(" ".join(fields) + "\n")


@SimtelTableWriter.register("atmospheric_transmission")
class AtmosphericTransmissionWriter:
    """Serialize atmospheric transmission matrices."""

    @staticmethod
    def write(table, output_path, contract):
        """Write altitude header and wavelength rows by explicit matrix lookup."""
        axis_0, axis_1 = contract["matrix_axes"]
        value_column = contract["value_column"]
        values = {(_scalar(row[axis_0]), _scalar(row[axis_1])): row[value_column] for row in table}
        axis_0_values = sorted(set(_raw_values(table[axis_0])))
        axis_1_values = sorted(set(_raw_values(table[axis_1])))
        level = _scalar(table.meta.get("observatory_level"))
        with output_path.open("w", encoding="utf-8") as file:
            header = "# H1= " + " ".join(_format(value, contract) for value in axis_1_values)
            file.write(
                f"# H2= {_format(level, contract)}, {header[2:]}\n"
                if level is not None
                else header + "\n"
            )
            for axis_0_value in axis_0_values:
                fields = [_format(axis_0_value, contract)]
                fields.extend(
                    _format(
                        values.get((axis_0_value, axis_1_value), contract.get("missing_value")),
                        contract,
                    )
                    for axis_1_value in axis_1_values
                )
                file.write(" ".join(fields) + "\n")


def _scalar(value):
    """Return a scalar value from an Astropy quantity."""
    return getattr(value, "value", value)
