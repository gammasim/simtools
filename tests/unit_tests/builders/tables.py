"""Build small, schema-valid tables for unit tests."""

from pathlib import Path

import numpy as np
import yaml
from astropy.table import Table

from simtools.constants import SCHEMA_PATH
from simtools.data_model.validate_data import DataValidator

REDUCED_EVENT_SCHEMAS = {
    "SHOWERS": "reduced_event_showers.schema.yml",
    "TRIGGERS": "reduced_event_triggers.schema.yml",
    "FILE_INFO": "reduced_event_file_info.schema.yml",
}
TRIGGER_HISTOGRAM_SCHEMAS = {
    "TRIGGER_HISTOGRAM_METADATA": "trigger_histogram_reference_metadata.schema.yml",
    "TRIGGER_HISTOGRAM_BINS": "trigger_histogram_reference_bins.schema.yml",
    "TRIGGER_TOPOLOGY_COUNTS": "trigger_histogram_topology_counts.schema.yml",
    "TRIGGER_SUBSET_HISTOGRAMS": "trigger_histogram_subset_histograms.schema.yml",
}


def reduced_event_tables(values=None, rows=1, seed=0):
    """Build schema-valid SHOWERS, TRIGGERS, and FILE_INFO tables."""
    return _build_tables(REDUCED_EVENT_SCHEMAS, values, rows, seed)


def trigger_histogram_tables(values=None, rows=1, seed=0):
    """Build schema-valid trigger-histogram tables."""
    return _build_tables(TRIGGER_HISTOGRAM_SCHEMAS, values, rows, seed)


def _build_tables(schemas, values, rows, seed):
    if rows < 1:
        raise ValueError("rows must be a positive integer")
    values = values or {}
    return {
        name: _build_table(SCHEMA_PATH / schema_name, values.get(name, {}), rows, seed)
        for name, schema_name in schemas.items()
    }


def _build_table(schema_file, overrides, rows, seed):
    with Path(schema_file).open(encoding="utf-8") as stream:
        definition = yaml.safe_load(stream)
    columns = {}
    rng = np.random.default_rng(seed)
    for column in definition["data"][0]["table_columns"]:
        name = column["name"]
        if name in overrides:
            data = np.asarray(overrides[name], dtype=np.dtype(column["type"]))
            if data.ndim == 0:
                data = np.repeat(data, rows)
        else:
            data = _default_values(name, column["type"], rows, rng)
        columns[name] = data
    table = Table(columns)
    for column in definition["data"][0]["table_columns"]:
        if column.get("unit"):
            table[column["name"]].unit = column["unit"]
    return DataValidator(schema_file=schema_file, data_table=table).validate_and_transform()


def _default_values(name, data_type, rows, rng):
    if name == "simulated_energy":
        values = _power_law_energies(rows, rng)
    elif name in {"x_core", "y_core"}:
        values = _disk_positions(rows, rng.bit_generator.seed_seq)[name]
    elif data_type == "string":
        values = ["test"] * rows
    elif data_type.startswith("float"):
        values = np.ones(rows)
    elif data_type.startswith("uint") or data_type.startswith("int"):
        values = np.arange(rows, dtype=np.dtype(data_type))
    else:
        raise ValueError(f"No table default is defined for column '{name}' ({data_type}).")
    return values


def _power_law_energies(rows, rng, e_min=0.1, e_max=100.0, index=2.0):
    """Draw energies [TeV] from dN/dE ~ E^-index by inverse-transform sampling."""
    exponent = 1.0 - index
    low, high = e_min**exponent, e_max**exponent
    return (low + rng.random(rows) * (high - low)) ** (1.0 / exponent)


def _disk_positions(rows, seed_sequence, radius=100.0):
    """Return core positions [m] uniform on a disk; x and y share the same draws."""
    rng = np.random.default_rng(seed_sequence)
    distance = radius * np.sqrt(rng.random(rows))
    angle = rng.uniform(0.0, 2.0 * np.pi, rows)
    return {"x_core": distance * np.cos(angle), "y_core": distance * np.sin(angle)}
