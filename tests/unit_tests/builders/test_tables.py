"""Tests for schema-driven test table builders."""

import numpy as np
import pytest

from tests.unit_tests.builders.tables import reduced_event_tables, trigger_histogram_tables


def test_reduced_event_tables_build_schema_columns():
    tables = reduced_event_tables(rows=3)

    assert set(tables) == {"SHOWERS", "TRIGGERS", "FILE_INFO"}
    assert len(tables["SHOWERS"]) == 3
    assert tables["SHOWERS"]["simulated_energy"].unit == "TeV"
    assert np.all(np.asarray(tables["SHOWERS"]["simulated_energy"]) > 0)


def test_reduced_event_tables_accept_column_values():
    tables = reduced_event_tables(
        values={"SHOWERS": {"shower_id": [12, 13], "simulated_energy": [1.0, 2.0]}},
        rows=2,
    )

    assert list(tables["SHOWERS"]["shower_id"]) == [12, 13]
    assert list(tables["SHOWERS"]["simulated_energy"]) == [1.0, 2.0]


def test_reduced_event_tables_seed_disk_positions():
    first = reduced_event_tables(rows=5, seed=17)["SHOWERS"]
    second = reduced_event_tables(rows=5, seed=17)["SHOWERS"]

    assert np.array_equal(first["x_core"], second["x_core"])
    assert np.array_equal(first["y_core"], second["y_core"])
    assert np.all(np.hypot(first["x_core"], first["y_core"]) <= 100.0)


def test_trigger_histogram_tables_build_all_schemas():
    tables = trigger_histogram_tables(rows=2)

    assert len(tables) == 4
    assert all(len(table) == 2 for table in tables.values())
    assert tables["TRIGGER_HISTOGRAM_BINS"]["energy_low"].unit == "TeV"


def test_table_builder_rejects_nonpositive_row_count():
    with pytest.raises(ValueError, match="rows must be a positive integer"):
        reduced_event_tables(rows=0)


def test_reduced_event_tables_cores_lie_on_disk_and_energies_follow_range():
    tables = reduced_event_tables(rows=200, seed=3)
    showers = tables["SHOWERS"]
    radius = np.hypot(np.asarray(showers["x_core"]), np.asarray(showers["y_core"]))
    energy = np.asarray(showers["simulated_energy"])

    assert np.all(radius <= 100.0)
    assert np.all((energy >= 0.1) & (energy <= 100.0))
    assert np.median(energy) < 1.0


@pytest.mark.parametrize("file_name", ["input.simtel.gz", ["a.simtel.gz", "b.simtel.gz"]])
def test_reduced_event_tables_accept_string_overrides(file_name):
    tables = reduced_event_tables(values={"FILE_INFO": {"file_name": file_name}}, rows=2)

    expected = file_name if isinstance(file_name, list) else [file_name, file_name]
    assert list(tables["FILE_INFO"]["file_name"]) == expected
