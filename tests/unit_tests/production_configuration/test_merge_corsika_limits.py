"""Unit tests for merge_corsika_limits.py module."""

from pathlib import Path
from unittest.mock import patch

import pytest
from astropy.table import Column, Table, vstack

from simtools.production_configuration.merge_corsika_limits import CorsikaMergeLimits

LIMITS_FILE_1 = "limits_1.ecsv"
LIMITS_FILE_2 = "limits_2.ecsv"
LIMITS_FILE_3 = "limits_3.ecsv"
ECSV_FORMAT = "ascii.ecsv"


def create_test_table(
    zenith,
    azimuth,
    nsb_level,
    array_name="test_array",
    telescope_ids=None,
    primary_particle="gamma",
    loss_fraction=1e-6,
    lower_energy_limit=0.01,
):
    """Create a test CORSIKA limits table with the given parameters."""
    if telescope_ids is None:
        telescope_ids = [1, 2, 3]

    columns = [
        Column(data=[primary_particle], name="primary_particle"),
        Column(data=[array_name], name="array_name"),
        Column(data=[telescope_ids], name="telescope_ids"),
        Column(data=[zenith], name="zenith"),
        Column(data=[azimuth], name="azimuth"),
        Column(data=[nsb_level], name="nsb_level"),
        Column(data=[lower_energy_limit], name="lower_energy_limit"),
        Column(data=[2000], name="upper_radius_limit"),
        Column(data=[10], name="viewcone_radius"),
        Column(data=[loss_fraction], name="loss_fraction"),
    ]

    table = Table(columns)
    table.meta = {
        "created": "2025-07-09T12:48:35.535873",
        "description": "Lookup table for CORSIKA limits computed from simulations.",
    }

    return table


@patch("simtools.production_configuration.merge_corsika_limits.data_reader.read_table_from_file")
def test_merge_tables(mock_read_table, tmp_path):
    """Test merging multiple CORSIKA limit tables."""
    table1 = create_test_table(20, 0, "dark", "layout1")
    table2 = create_test_table(40, 0, "dark", "layout1")
    table3 = create_test_table(20, 180, "moon", "layout2")
    mock_read_table.side_effect = [table1, table2, table3]

    merger = CorsikaMergeLimits(output_dir=tmp_path)
    input_files = [LIMITS_FILE_1, LIMITS_FILE_2, LIMITS_FILE_3]
    merged_table = merger.merge_tables(input_files)

    assert len(merged_table) == 3

    assert merged_table[0]["array_name"] == "layout1"
    assert merged_table[1]["array_name"] == "layout1"
    assert merged_table[2]["array_name"] == "layout2"

    # Check loss_fraction is now in columns, not metadata
    assert "loss_fraction" in merged_table.colnames
    assert merged_table["loss_fraction"][0] == 1e-6


@patch("simtools.production_configuration.merge_corsika_limits.data_reader.read_table_from_file")
def test_merge_tables_with_duplicates(mock_read_table, tmp_path):
    """Test merging CORSIKA limit tables with duplicate grid points."""
    # Create test tables with a duplicate grid point
    table1 = create_test_table(20, 0, "dark", "layout1")
    table2 = create_test_table(40, 0, "dark", "layout1")

    # Create a duplicate of the first grid point with IDENTICAL values
    duplicate = create_test_table(20, 0, "dark", "layout1")
    # No change to lower_energy_limit to ensure values are consistent
    mock_read_table.side_effect = [table1, table2, duplicate]

    merger = CorsikaMergeLimits(output_dir=tmp_path)
    input_files = [LIMITS_FILE_1, LIMITS_FILE_2, LIMITS_FILE_3]
    merged_table = merger.merge_tables(input_files)

    assert len(merged_table) == 2  # Duplicate row ignored

    # Find the duplicate row
    mask = (
        (merged_table["zenith"] == 20)
        & (merged_table["azimuth"] == 0)
        & (merged_table["nsb_level"] == "dark")
    )
    duplicate_rows = merged_table[mask]

    # Should find the row
    assert len(duplicate_rows) == 1

    # Check that the value is correct
    assert duplicate_rows["lower_energy_limit"][0] == 0.01


@patch("simtools.production_configuration.merge_corsika_limits.data_reader.read_table_from_file")
def test_merge_tables_different_loss_fractions(mock_read_table, tmp_path):
    """Test merging tables with different loss fraction values."""
    # Create test tables with different loss fractions
    table1 = create_test_table(20, 0, "dark", "layout1", loss_fraction=1e-6)
    table2 = create_test_table(40, 0, "dark", "layout1", loss_fraction=2e-6)
    mock_read_table.side_effect = [table1, table2]

    merger = CorsikaMergeLimits(output_dir=tmp_path)
    input_files = [LIMITS_FILE_1, LIMITS_FILE_2]
    merged_table = merger.merge_tables(input_files)

    # Check that both loss_fraction values are preserved in the columns
    assert "loss_fraction" in merged_table.colnames
    assert merged_table["loss_fraction"][0] == 1e-6
    assert merged_table["loss_fraction"][1] == 2e-6


def test_check_grid_completeness(tmp_path):
    """Test checking grid completeness."""
    # Create test tables
    tables = [
        create_test_table(20, 0, "dark", "layout1"),
        create_test_table(40, 0, "dark", "layout1"),
        create_test_table(60, 0, "dark", "layout1"),
        create_test_table(20, 180, "dark", "layout1"),
        create_test_table(40, 180, "dark", "layout1"),
        # Missing 60, 180, dark, layout1
    ]
    merged_table = vstack(tables)
    grid_definition = {
        "zenith": [20, 40, 60],
        "azimuth": [0, 180],
        "nsb_level": ["dark"],
        "array_names": ["layout1"],  # Changed from layouts to array_names
    }
    merger = CorsikaMergeLimits(output_dir=tmp_path)

    # Test with incomplete grid
    is_complete, result = merger.check_grid_completeness(merged_table, grid_definition)
    assert not is_complete
    assert result["expected"] == 6
    assert result["found"] == 5
    assert len(result["missing"]) == 1

    # Test with complete grid
    tables.append(create_test_table(60, 180, "dark", "layout1"))
    complete_table = vstack(tables)
    is_complete, result = merger.check_grid_completeness(complete_table, grid_definition)
    assert is_complete
    assert result["expected"] == 6
    assert result["found"] == 6
    assert len(result["missing"]) == 0

    # Test with no grid definition
    is_complete, result = merger.check_grid_completeness(complete_table, None)
    assert is_complete
    assert not result


@patch("matplotlib.pyplot.savefig")
def test_plot_grid_coverage(mock_savefig, tmp_path):
    """Test generating grid coverage plots."""
    table = vstack(
        [
            create_test_table(20, 0, "dark", "layout1"),
            create_test_table(40, 0, "dark", "layout1"),
        ]
    )
    grid_definition = {
        "zenith": [20, 40],
        "azimuth": [0],
        "nsb_level": ["dark"],
        "array_name": ["layout1"],
    }
    merger = CorsikaMergeLimits(output_dir=tmp_path)

    # Test plotting with grid definition
    output_files = merger.plot_grid_coverage(table, grid_definition)
    assert len(output_files) == 1
    mock_savefig.assert_called_once()

    # Test plotting without grid definition
    mock_savefig.reset_mock()
    output_files = merger.plot_grid_coverage(table, None)
    assert not output_files
    mock_savefig.assert_not_called()


@patch("matplotlib.pyplot.savefig")
def test_plot_limits(mock_savefig, tmp_path):
    """Test generating limit plots."""
    table = vstack(
        [
            create_test_table(20, 0, "dark", "layout1"),
            create_test_table(40, 0, "dark", "layout1"),
            create_test_table(20, 0, "moon", "layout1"),
        ]
    )
    merger = CorsikaMergeLimits(output_dir=tmp_path)
    output_files = merger.plot_limits(table)

    assert len(output_files) == 1
    mock_savefig.assert_called_once()


def test_write_merged_table(tmp_path):
    """Test writing the merged table to file."""
    table = create_test_table(20, 0, "dark", "layout1")
    merger = CorsikaMergeLimits(output_dir=tmp_path)
    output_file = tmp_path / "merged_limits.ecsv"
    input_files = [LIMITS_FILE_1, LIMITS_FILE_2]
    grid_completeness = {
        "is_complete": True,
        "missing": [],
        "expected": 1,
        "found": 1,
    }

    with (
        patch("astropy.table.Table.write") as mock_write,
        patch(
            "simtools.production_configuration.merge_corsika_limits.MetadataCollector.dump"
        ) as mock_dump,
    ):
        merger.write_merged_table(table, output_file, input_files, grid_completeness)

        mock_write.assert_called_once_with(output_file, format=ECSV_FORMAT, overwrite=True)
        mock_dump.assert_called_once()


def test_read_file_list(tmp_path):
    """Test reading a list of files from a text file."""
    # Create a test file with a list of files
    file_list_path = tmp_path / "file_list.txt"
    with open(file_list_path, "w", encoding="utf-8") as f:
        f.write("# This is a comment\n")
        f.write("\n")  # Empty line
        f.write(f"{tmp_path}/file1.ecsv\n")
        f.write(f"{tmp_path}/file2.ecsv\n")
        f.write("  # Another comment\n")
        f.write(f"{tmp_path}/file3.ecsv  \n")  # With trailing whitespace

    merger = CorsikaMergeLimits(output_dir=tmp_path)
    files = merger.read_file_list(file_list_path)

    assert len(files) == 3
    assert files[0] == Path(f"{tmp_path}/file1.ecsv")
    assert files[1] == Path(f"{tmp_path}/file2.ecsv")
    assert files[2] == Path(f"{tmp_path}/file3.ecsv")

    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        merger.read_file_list(tmp_path / "non_existent.txt")


@patch("simtools.production_configuration.merge_corsika_limits.data_reader.read_table_from_file")
def test_merge_tables_with_inconsistent_duplicates(mock_read_table, tmp_path):
    """Test merging tables with inconsistent duplicate grid points."""
    table1 = create_test_table(20, 0, "dark", "layout1", lower_energy_limit=0.01)
    # Duplicate with a different value
    table2 = create_test_table(20, 0, "dark", "layout1", lower_energy_limit=0.02)
    mock_read_table.side_effect = [table1, table2]

    merger = CorsikaMergeLimits(output_dir=tmp_path)
    input_files = [LIMITS_FILE_1, LIMITS_FILE_2]

    # Should always raise an error for inconsistent values
    with pytest.raises(ValueError, match="Found 1 grid points with inconsistent values"):
        merger.merge_tables(input_files)


@patch("simtools.production_configuration.merge_corsika_limits.data_reader.read_table_from_file")
def test_merge_tables_with_consistent_duplicates(mock_read_table, tmp_path):
    """Test merging tables with consistent duplicate grid points."""
    table1 = create_test_table(20, 0, "dark", "layout1")
    # Exact same table
    table2 = create_test_table(20, 0, "dark", "layout1")
    mock_read_table.side_effect = [table1, table2]

    merger = CorsikaMergeLimits(output_dir=tmp_path)
    input_files = [LIMITS_FILE_1, LIMITS_FILE_2]

    # Should merge without error
    merged_table = merger.merge_tables(input_files)
    assert len(merged_table) == 1
