"""Unit tests for merge_corsika_limits.py module."""

from unittest.mock import patch

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
        Column(data=[0.01], name="lower_energy_limit"),
        Column(data=[2000], name="upper_radius_limit"),
        Column(data=[10], name="viewcone_radius"),
    ]

    table = Table(columns)
    table.meta = {
        "created": "2025-07-09T12:48:35.535873",
        "description": "Lookup table for CORSIKA limits computed from simulations.",
        "loss_fraction": loss_fraction,
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

    assert "loss_fraction" in merged_table.meta
    assert merged_table.meta["loss_fraction"] == 1e-6


@patch("simtools.production_configuration.merge_corsika_limits.data_reader.read_table_from_file")
def test_merge_tables_with_duplicates(mock_read_table, tmp_path):
    """Test merging CORSIKA limit tables with duplicate grid points."""
    # Create test tables with a duplicate grid point
    table1 = create_test_table(20, 0, "dark", "layout1")
    table2 = create_test_table(40, 0, "dark", "layout1")

    # Create a duplicate of the first grid point with different values
    duplicate = create_test_table(20, 0, "dark", "layout1")
    duplicate["lower_energy_limit"] = 0.02  # Different value
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

    # Check that the last value was kept (from the duplicate with energy 0.02)
    assert duplicate_rows["lower_energy_limit"][0] == 0.02


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

    # Check that metadata from first table was used
    assert merged_table.meta["loss_fraction"] == 1e-6


def test_check_grid_completeness():
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
        "layouts": ["layout1"],
    }

    merger = CorsikaMergeLimits(output_dir="/tmp")
    is_complete, result = merger.check_grid_completeness(merged_table, grid_definition)

    # Should not be complete
    assert not is_complete
    assert result["expected"] == 6
    assert result["found"] == 5
    assert len(result["missing"]) == 1

    # Add missing point and check again
    tables.append(create_test_table(60, 180, "dark", "layout1"))
    complete_table = vstack(tables)

    is_complete, result = merger.check_grid_completeness(complete_table, grid_definition)

    # Should be complete now
    assert is_complete
    assert result["expected"] == 6
    assert result["found"] == 6
    assert len(result["missing"]) == 0


@patch("matplotlib.pyplot.savefig")
def test_plot_grid_coverage(mock_savefig, tmp_path):
    """Test generating grid coverage plots."""
    table = vstack(
        [
            create_test_table(20, 0, "dark", "layout1"),
            create_test_table(40, 0, "dark", "layout1"),
        ]
    )
    merger = CorsikaMergeLimits(output_dir=tmp_path)
    output_files = merger.plot_grid_coverage(table)

    assert len(output_files) == 1
    mock_savefig.assert_called_once()


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
