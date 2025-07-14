"""Unit tests for production_merge_corsika_limits."""

import numpy as np
from astropy.table import Column, Table, vstack

from simtools.production_configuration.merge_corsika_limits import CorsikaMergeLimits


def create_test_table(
    zenith,
    azimuth,
    nsb_level,
    array_name="test_array",
    telescope_ids=None,
    primary_particle="gamma",
):
    """Create a test CORSIKA limits table with the given parameters."""
    if telescope_ids is None:
        telescope_ids = [1, 2, 3]

    columns = [
        Column(data=[primary_particle], name="primary_particle"),
        Column(data=[array_name], name="array_name"),  # Use array_name instead of layout
        Column(data=[telescope_ids], name="telescope_ids"),
        Column(data=[zenith], name="zenith"),
        Column(data=[azimuth], name="azimuth"),
        Column(data=[nsb_level], name="nsb_level"),
        Column(data=[0.01], name="lower_energy_limit"),
        Column(data=[2000], name="upper_radius_limit"),
        Column(data=[10], name="viewcone_radius"),
    ]

    table = Table(columns)
    table.meta = {
        "created": "2025-07-09T12:48:35.535873",
        "description": "Lookup table for CORSIKA limits computed from simulations.",
        "loss_fraction": 1e-6,
    }

    return table


def test_merge_corsika_limit_tables(tmp_path):
    """Test merging multiple CORSIKA limit tables."""
    # Create test tables
    table1 = create_test_table(20, 0, "dark")
    table2 = create_test_table(40, 0, "dark")
    table3 = create_test_table(20, 180, "moon")

    # Write tables to files
    file1 = tmp_path / "limits_1.ecsv"
    file2 = tmp_path / "limits_2.ecsv"
    file3 = tmp_path / "limits_3.ecsv"

    table1.write(file1, format="ascii.ecsv")
    table2.write(file2, format="ascii.ecsv")
    table3.write(file3, format="ascii.ecsv")

    # Merge tables
    input_files = [file1, file2, file3]
    merger = CorsikaMergeLimits(output_dir=tmp_path)
    merged_table = merger.merge_tables(input_files)

    # Check results
    assert len(merged_table) == 3
    assert np.all(merged_table["zenith"] == [20, 40, 20])
    assert np.all(merged_table["azimuth"] == [0, 0, 180])
    assert np.all(merged_table["nsb_level"] == ["dark", "dark", "moon"])
    assert "loss_fraction" in merged_table.meta


def test_check_grid_completeness():
    """Test checking grid completeness."""
    # Create test table with some grid points
    tables = [
        create_test_table(20, 0, "dark", "layout1"),
        create_test_table(40, 0, "dark", "layout1"),
        create_test_table(60, 0, "dark", "layout1"),
        create_test_table(20, 180, "dark", "layout1"),
        create_test_table(40, 180, "dark", "layout1"),
        # Missing 60, 180, dark, layout1
        create_test_table(20, 0, "moon", "layout1"),
        create_test_table(40, 0, "moon", "layout1"),
        create_test_table(60, 0, "moon", "layout1"),
        # Missing 20, 180, moon, layout1
        create_test_table(40, 180, "moon", "layout1"),
        create_test_table(60, 180, "moon", "layout1"),
    ]

    merged_table = vstack(tables)

    # Define expected grid
    grid_definition = {
        "zenith": [20, 40, 60],
        "azimuth": [0, 180],
        "nsb_level": ["dark", "moon"],
        "layouts": ["layout1"],
    }

    # Check completeness
    merger = CorsikaMergeLimits(output_dir="/tmp")
    is_complete, result = merger.check_grid_completeness(merged_table, grid_definition)

    assert not is_complete
    assert result["expected"] == 12
    assert result["found"] == 10
    assert len(result["missing"]) == 2


def test_check_grid_completeness_complete():
    """Test checking grid completeness with a complete grid."""
    # Create test table with all grid points
    grid_points = [
        (20, 0, "dark", "layout1"),
        (40, 0, "dark", "layout1"),
        (60, 0, "dark", "layout1"),
        (20, 180, "dark", "layout1"),
        (40, 180, "dark", "layout1"),
        (60, 180, "dark", "layout1"),
    ]

    tables = [create_test_table(*point) for point in grid_points]
    merged_table = Table.vstack(tables)

    # Define expected grid
    grid_definition = {
        "zenith": [20, 40, 60],
        "azimuth": [0, 180],
        "nsb_level": ["dark"],
        "layouts": ["layout1"],
    }

    # Check completeness
    merger = CorsikaMergeLimits(output_dir="/tmp")
    is_complete, result = merger.check_grid_completeness(merged_table, grid_definition)

    assert is_complete
    assert result["expected"] == 6
    assert result["found"] == 6
    assert len(result["missing"]) == 0


def test_check_grid_completeness_auto_extract():
    """Test that grid definition is automatically extracted if not provided."""
    # Create test table with some grid points
    grid_points = [
        (20, 0, "dark", "layout1"),
        (40, 0, "dark", "layout1"),
        (60, 0, "dark", "layout1"),
    ]

    tables = [create_test_table(*point) for point in grid_points]
    merged_table = Table.vstack(tables)

    # Check completeness without providing grid definition
    merger = CorsikaMergeLimits(output_dir="/tmp")
    is_complete, result = merger.check_grid_completeness(merged_table)

    # It should be complete since we're extracting the grid from the table
    assert is_complete
    assert result["expected"] == 3
    assert result["found"] == 3
    assert len(result["missing"]) == 0


def test_different_column_names():
    """Test handling of different column naming conventions."""
    # Create a table with 'array_name' instead of 'layout' and numeric nsb instead of string nsb_level
    columns1 = [
        Column(data=["gamma"], name="primary_particle"),
        Column(data=["alpha"], name="array_name"),  # Using array_name instead of layout
        Column(
            data=[
                [
                    "LSTN-01",
                    "LSTN-02",
                    "LSTN-03",
                    "LSTN-04",
                    "MSTN-01",
                    "MSTN-02",
                    "MSTN-03",
                    "MSTN-04",
                ]
            ],
            name="telescope_ids",
        ),
        Column(data=[20], name="zenith"),
        Column(data=[0], name="azimuth"),
        Column(data=[1.0], name="nsb"),  # Using nsb as float instead of nsb_level string
        Column(data=[0.01], name="lower_energy_limit"),
        Column(data=[1500], name="upper_radius_limit"),
        Column(data=[10], name="viewcone_radius"),
    ]

    # Create a table with standard column names
    columns2 = [
        Column(data=["gamma"], name="primary_particle"),
        Column(data=["beta"], name="layout"),  # Using layout instead of array_name
        Column(data=[["MSTN-01", "MSTN-02", "MSTN-03", "MSTN-04"]], name="telescope_ids"),
        Column(data=[40], name="zenith"),
        Column(data=[0], name="azimuth"),
        Column(data=["dark"], name="nsb_level"),  # Using nsb_level as string
        Column(data=[0.02], name="lower_energy_limit"),
        Column(data=[2000], name="upper_radius_limit"),
        Column(data=[8], name="viewcone_radius"),
    ]

    table1 = Table(columns1)
    table2 = Table(columns2)

    # Merge the tables
    merged = Table.vstack([table1, table2])

    # Test grid completeness check
    merger = CorsikaMergeLimits(output_dir="/tmp")
    is_complete, results = merger.check_grid_completeness(merged)

    # We should find 2 combinations (all that exist in the table)
    assert is_complete
    assert results["expected"] == 2
    assert results["found"] == 2

    # Check that we can still access the data despite different column names
    nsb_column = "nsb_level" if "nsb_level" in merged.colnames else "nsb"
    layout_column = "layout" if "layout" in merged.colnames else "array_name"

    # The merged table should have both rows
    assert len(merged) == 2

    # We should be able to access the nsb values
    nsb_values = merged[nsb_column]
    assert len(nsb_values) == 2

    # We should be able to access the layout values
    layout_values = merged[layout_column]
    assert len(layout_values) == 2
    assert "alpha" in layout_values
    assert "beta" in layout_values


def test_nsb_type_conversion():
    """Test handling of different NSB types (numeric vs string)."""
    # Create a table with NSB as a float
    table1 = create_test_table(20, 0, 1.0, "alpha")

    # Create a table with NSB as a string
    table2 = create_test_table(40, 0, "1.0", "beta")

    # Merge tables
    merged = Table.vstack([table1, table2])

    # Define the expected grid with numeric NSB
    grid_definition = {
        "zenith": [20, 40],
        "azimuth": [0],
        "nsb_level": [1.0],
        "layouts": ["alpha", "beta"],
    }

    # Check grid completeness - should find both entries despite type differences
    merger = CorsikaMergeLimits(output_dir="/tmp")
    is_complete, results = merger.check_grid_completeness(merged, grid_definition)

    assert results["expected"] == 4  # 2 zeniths x 1 azimuth x 1 nsb x 2 layouts
    assert results["found"] >= 2  # At least the two combinations we explicitly created

    # Now try with string NSB in the grid definition
    grid_definition = {
        "zenith": [20, 40],
        "azimuth": [0],
        "nsb_level": ["1.0"],
        "layouts": ["alpha", "beta"],
    }

    # Check grid completeness again
    merger = CorsikaMergeLimits(output_dir="/tmp")
    is_complete, results = merger.check_grid_completeness(merged, grid_definition)

    assert results["expected"] == 4
    assert results["found"] >= 2  # Should still find at least our two rows
