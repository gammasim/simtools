"""Unit tests for merge_corsika_limits.py module."""

from astropy.table import Column, Table

from simtools.production_configuration.merge_corsika_limits import CorsikaMergeLimits


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


def test_merge_tables(tmp_path):
    """Test merging multiple CORSIKA limit tables."""
    table1 = create_test_table(20, 0, "dark", "layout1")
    table2 = create_test_table(40, 0, "dark", "layout1")
    table3 = create_test_table(20, 180, "moon", "layout2")

    file1 = tmp_path / "limits_1.ecsv"
    file2 = tmp_path / "limits_2.ecsv"
    file3 = tmp_path / "limits_3.ecsv"

    table1.write(file1, format="ascii.ecsv")
    table2.write(file2, format="ascii.ecsv")
    table3.write(file3, format="ascii.ecsv")

    merger = CorsikaMergeLimits()
    input_files = [file1, file2, file3]
    merged_table = merger.merge_tables(input_files)

    assert len(merged_table) == 3

    assert merged_table[0]["array_name"] == "layout1"
    assert merged_table[1]["array_name"] == "layout1"
    assert merged_table[2]["array_name"] == "layout2"

    assert "loss_fraction" in merged_table.meta
    assert merged_table.meta["loss_fraction"] == 1e-6


def test_merge_tables_with_duplicates(tmp_path):
    """Test merging CORSIKA limit tables with duplicate grid points."""
    # Create test tables with a duplicate grid point
    table1 = create_test_table(20, 0, "dark", "layout1", telescope_ids=[1, 2, 3])
    table2 = create_test_table(40, 0, "dark", "layout1", telescope_ids=[4, 5, 6])

    # Create a duplicate of the first grid point with different values
    duplicate = create_test_table(20, 0, "dark", "layout1", telescope_ids=[7, 8, 9])
    duplicate["lower_energy_limit"] = 0.02  # Different value

    file1 = tmp_path / "limits_1.ecsv"
    file2 = tmp_path / "limits_2.ecsv"
    file3 = tmp_path / "limits_3.ecsv"

    table1.write(file1, format="ascii.ecsv")
    table2.write(file2, format="ascii.ecsv")
    duplicate.write(file3, format="ascii.ecsv")

    merger = CorsikaMergeLimits()
    input_files = [file1, file2, file3]
    merged_table = merger.merge_tables(input_files)

    assert len(merged_table) == 3

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


def test_merge_tables_different_loss_fractions(tmp_path):
    """Test merging tables with different loss fraction values."""
    # Create test tables with different loss fractions
    table1 = create_test_table(20, 0, "dark", "layout1", loss_fraction=1e-6)
    table2 = create_test_table(40, 0, "dark", "layout1", loss_fraction=2e-6)

    file1 = tmp_path / "limits_1.ecsv"
    file2 = tmp_path / "limits_2.ecsv"

    table1.write(file1, format="ascii.ecsv")
    table2.write(file2, format="ascii.ecsv")

    merger = CorsikaMergeLimits()
    input_files = [file1, file2]
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

    merged_table = Table.vstack(tables)

    grid_definition = {
        "zenith": [20, 40, 60],
        "azimuth": [0, 180],
        "nsb_level": ["dark"],
        "layouts": ["layout1"],
    }

    merger = CorsikaMergeLimits()
    is_complete, result = merger.check_grid_completeness(merged_table, grid_definition)

    # Should not be complete
    assert not is_complete
    assert result["expected"] == 6
    assert result["found"] == 5
    assert len(result["missing"]) == 1

    # Add missing point and check again
    tables.append(create_test_table(60, 180, "dark", "layout1"))
    complete_table = Table.vstack(tables)

    is_complete, result = merger.check_grid_completeness(complete_table, grid_definition)

    # Should be complete now
    assert is_complete
    assert result["expected"] == 6
    assert result["found"] == 6
    assert len(result["missing"]) == 0
